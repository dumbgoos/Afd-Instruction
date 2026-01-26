import json
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
import os
import argparse
from dotenv import load_dotenv


def build_messages_icl(question: str):
    system_prompt = (
        "You are a precise scientific assistant.\n"
        "Return the answer in the SAME concise declarative style as the ground_truth labels in this dataset.\n"
        "- Output EXACTLY ONE short sentence or phrase.\n"
        "- NO explanation, NO reasoning, NO extra words.\n"
        "- NO preface like 'the answer is'.\n"
        "- Keep it factual and terse. If truly unknowable, answer 'unknown'."
    )

    icl_q1 = ("<H>VQLVQSGAEVKKPGASVKVSCKASGYIFSDYNIHWVRQAPGQGLEWMGWISPDSDDTNYAQSFQGRVTMTRDTSITTVYMELSSLRSDDTAVYFCARSVGYCSLNSCQRWMWFDTWGQGALVTVSSA</H> "
              "<L>PVLTQPPSASGPPGQSVSISCSGSRSNIGTNFVYWYQQLPGAAPKLLIYKNDQRPSGVPERFFGSKSGTSASLAISGLRSEDEVDYYCAAWDDSLSGHVFGAGTKVTVLGQ</L> "
              "Why is this antibody resilient against common escape mutations?")
    icl_a1 = "Its binding to a non-canonical epitope confers resilience against such mutations."

    icl_q2 = ("<H>QVQLVQSGAEVKKPGASVKVSCKASGYKFTGFVMHWVRQAPGQGLEWMGFINPYNDDIQSNERFRGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGNGYNFDGAYRFFDFWGQGTMVTVSSA</H> "
              "<L>DIVMTQSPLSLPVTPGEPASISCRSSQRLVHSNGNTYLHWYLQKPGQSPRLLIYRVSNRFPGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCSQSTHVPYTFGQGTKLEIKRT</L> "
              "Which residues on TRBC1 does this antibody interact with?")
    icl_a2 = "It interacts with Asn119 and Lys120."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": icl_q1},
        {"role": "assistant", "content": icl_a1},
        {"role": "user", "content": icl_q2},
        {"role": "assistant", "content": icl_a2},
        {"role": "user", "content": question}
    ]


def run_inference_icl(input_path: str, output_csv: str, api_key: str, base_url: str, model: str):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=["pdb_id", "query", "ground_truth", "prediction", "correct"]).to_csv(output_csv, index=False)

    correct_count = 0
    processed_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    with open(input_path, 'r', encoding='utf-8') as f, tqdm(f, total=total, desc="Processing with ICL", unit="rec") as pbar:
        for line in pbar:
            item = json.loads(line)
            pdb_id = item.get("pdb_id", f"item_{processed_count}")
            question = next(m["value"] for m in item["messages"] if m["from"] == "human")
            expected = next(m["value"].strip().lower() for m in item["messages"] if m["from"] == "gpt")

            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=build_messages_icl(question),
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=64
                )
                answer = resp.choices[0].message.content.strip()
                answer_lc = answer.lower()
            except Exception as e:
                print(f"Error processing item {processed_count}: {e}")
                answer = "error"
                answer_lc = "error"

            correct = (answer_lc == expected)
            processed_count += 1
            correct_count += int(correct)

            row = {
                "pdb_id": pdb_id,
                "query": question,
                "ground_truth": expected,
                "prediction": answer_lc,
                "correct": correct
            }

            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)

            acc = correct_count / processed_count if processed_count > 0 else 0
            pbar.set_postfix(acc=f"{acc:.2%}")

    final_acc = correct_count / processed_count if processed_count > 0 else 0
    print(f"\nFinal Accuracy (ICL): {final_acc:.2%}")
    return final_acc


def parse_args():
    parser = argparse.ArgumentParser(description="API-based inference with ICL")
    parser.add_argument("--input", type=str, default="../../data/toy.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="./results_icl.csv", help="Output CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv("config.env")
    args = parse_args()
    
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")
    
    run_inference_icl(
        input_path=args.input,
        output_csv=args.output,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
