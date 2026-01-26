from typing import Dict, Any
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template


def prepare_tokenizer_with_template(tokenizer, chat_template_cfg: Dict[str, Any]):
    tokenizer = get_chat_template(
        tokenizer,
        mapping=chat_template_cfg.get("mapping", {"role": "from", "content": "value", "user": "human", "assistant": "gpt"}),
        chat_template=chat_template_cfg.get("template", "chatml"),
    )
    return tokenizer


def apply_template_factory(tokenizer):
    def apply_template(examples):
        messages = examples["messages"]
        text = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        return {"text": text}
    return apply_template


def load_and_prepare_dataset(data_file: str, tokenizer) -> Any:
    dataset = load_dataset("json", data_files=data_file, split="train")
    apply_fn = apply_template_factory(tokenizer)
    dataset = dataset.map(apply_fn, batched=True)
    return dataset