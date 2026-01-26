import os
import json
import csv
import re
import torch
import argparse
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, pipeline
from transformers import logging as hf_logging


def build_prompt_icl(question):
    system_prompt = (
        "You are a precise scientific assistant.\n"
        "Return the answer in the SAME concise declarative style as the ground_truth labels in this dataset.\n"
        "- Output EXACTLY ONE short phrase.\n"
        "- No explanation. No extra words. No preface.\n"
    )
    icl_q1 = ("<H>VQLVQSGAEVKKPGASVKVSCKASGYIFSDYNIHWVRQAPGQGLEWMGWISPDSDDTNYAQSFQGRVTMTRDTSITTVYMELSSLRSDDTAVYFCARSVGYCSLNSCQRWMWFDTWGQGALVTVSSA</H> "
              "<L>PVLTQPPSASGPPGQSVSISCSGSRSNIGTNFVYWYQQLPGAAPKLLIYKNDQRPSGVPERFFGSKSGTSASLAISGLRSEDEVDYYCAAWDDSLSGHVFGAGTKVTVLGQ</L> "
              "Why is this antibody resilient against common escape mutations?")
    icl_a1 = "binding to a non-canonical epitope"

    icl_q2 = ("<H>QVQLVQSGAEVKKPGASVKVSCKASGYKFTGFVMHWVRQAPGQGLEWMGFINPYNDDIQSNERFRGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGNGYNFDGAYRFFDFWGQGTMVTVSSA</H> "
              "<L>DIVMTQSPLSLPVTPGEPASISCRSSQRLVHSNGNTYLHWYLQKPGQSPRLLIYRVSNRFPGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCSQSTHVPYTFGQGTKLEIKRT</L> "
              "Which residues on TRBC1 does this antibody interact with?")
    icl_a2 = "Asn119 and Lys120"

    prompt = (
        f"{system_prompt}\n\n"
        f"Q: {icl_q1}\nA: {icl_a1}\n\n"
        f"Q: {icl_q2}\nA: {icl_a2}\n\n"
        f"Q: {question}\nA:"
    )
    return prompt


def build_prompt_no_icl(question):
    system_prompt = (
        "You are a precise scientific assistant specializing in antibody analysis.\n"
        "Answer the question about the given antibody sequence concisely and accurately.\n"
        "- Provide a direct, factual answer\n"
        "- Keep responses brief and to the point\n"
        "- If uncertain, answer 'unknown'"
    )
    return f"{system_prompt}\n\nQ: {question}\nA:"


def normalize_phrase(text):
    if not text:
        return "error"
    line = text.splitlines()[0].strip()
    line = re.sub(r'^[\'"""''`]+|[\'"""''`]+$', "", line)
    line = re.sub(r'[ \t]*[.。!！?？:：;；]+$', "", line)
    line = re.sub(r"\s+", " ", line)
    return line if line else "error"


def safe_get(items, role, default=""):
    for m in items:
        if m.get("from") == role:
            return m.get("value", "").strip()
    return default


def add_prefix_to_amino_acids(protein_sequence):
    return ''.join(f'<p>{aa}' for aa in protein_sequence)


def parse_question(question):
    m = re.match(r"<H>(?P<H>[^<]+)</H>\s*[, ]?\s*<L>(?P<L>[^<]+)</L>\s*(?P<Q>.+)", question)
    if not m:
        raise ValueError(f"Cannot parse question: {question!r}")
    return m.group("H"), m.group("L"), m.group("Q").strip()


def build_biot5_input(heavy, light, question_input):
    raw_protein = heavy + light
    protein_input = add_prefix_to_amino_acids(raw_protein)
    
    task_definition = (
        "You are a precise scientific assistant.\n"
        "Return the answer in the SAME concise declarative style as the ground_truth labels in this dataset.\n"
        "- Output EXACTLY ONE short phrase.\n"
        "- No explanation. No extra words. No preface.\n"
    )

    task_input = (
        f"{question_input} -\n"
        f"Input: Protein: <bop>{protein_input}<eop>\n"
        "Output: "
    )
    
    return task_definition + task_input


def run_inference(model_path, input_path, output_csv, model_type="transformers", use_icl=True, cuda_device="0", 
                 max_new_tokens=128, temperature=0.0, top_p=1.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    hf_logging.set_verbosity_error()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    
    if model_type == "biot5":
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        pipe = None
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.pad_token_id = pad_id
        model.config.eos_token_id = tokenizer.eos_token_id

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    need_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8", buffering=1) as fout:
        writer = csv.writer(fout)
        if need_header:
            writer.writerow(["pdb_id", "query", "ground_truth", "prediction", "correct"])

        total = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
        correct_cnt, processed = 0, 0

        with open(input_path, "r", encoding="utf-8") as fin:
            mode_str = "ICL" if use_icl else "No-ICL"
            pbar = tqdm(fin, total=total, desc=f"Processing {mode_str}", unit="rec")

            for line in pbar:
                item = json.loads(line)
                pdb_id = item.get("pdb_id", f"item_{processed}")
                question = safe_get(item["messages"], "human")
                expected = safe_get(item["messages"], "gpt").lower()

                try:
                    if model_type == "biot5":
                        heavy, light, question_input = parse_question(question)
                        model_input = build_biot5_input(heavy, light, question_input)
                        
                        input_ids = tokenizer(model_input, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids,
                                max_length=16,
                                num_beams=2
                            )
                        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    else:
                        if use_icl:
                            prompt = build_prompt_icl(question)
                        else:
                            prompt = build_prompt_no_icl(question)

                        outputs = pipe(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            top_p=top_p,
                            temperature=temperature,
                            return_full_text=False,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=pad_id,
                        )

                        gen = outputs[0]["generated_text"] if outputs and "generated_text" in outputs[0] else ""
                        pred = normalize_phrase(gen)
                except Exception as e:
                    print(f"Error processing item {processed}: {e}")
                    pred = "error"

                is_correct = (pred.lower() == expected) if expected != "" else False
                correct_cnt += int(is_correct)
                processed += 1

                writer.writerow([pdb_id, question, expected, pred, is_correct])
                fout.flush()

                acc = correct_cnt / processed if processed else 0.0
                pbar.set_postfix(acc=f"{acc:.2%}")

    final_acc = correct_cnt / processed if processed else 0.0
    mode_str = "ICL" if use_icl else "No-ICL"
    print(f"\nFinal Accuracy ({mode_str}): {final_acc:.2%}")
    return final_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Local transformers model inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_type", type=str, choices=["transformers", "biot5"], default="transformers", help="Model type")
    parser.add_argument("--input", type=str, default="../../data/toy.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="./local_results.csv", help="Output CSV file")
    parser.add_argument("--use_icl", action="store_true", help="Use in-context learning")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device ID")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        input_path=args.input,
        output_csv=args.output,
        model_type=args.model_type,
        use_icl=args.use_icl,
        cuda_device=args.cuda_device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
