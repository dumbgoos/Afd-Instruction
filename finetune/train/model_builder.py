from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
from unsloth import FastLanguageModel, is_bfloat16_supported


def build_model_and_tokenizer(paths: Dict[str, Any], max_seq_length: int, chat_template_cfg: Dict[str, Any],
                              unsloth_cfg: Dict[str, Any], lora_cfg: Dict[str, Any]):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=paths["base_model"],
        max_seq_length=max_seq_length,
        load_in_4bit=unsloth_cfg.get("load_in_4bit", True),
        dtype=unsloth_cfg.get("dtype", None),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        target_modules=lora_cfg.get("target_modules", ["q_proj","k_proj","v_proj","up_proj","down_proj","o_proj","gate_proj"]),
        use_rslora=lora_cfg.get("use_rslora", True),
        use_gradient_checkpointing=unsloth_cfg.get("use_gradient_checkpointing", "unsloth"),
    )

    return model, tokenizer


def fp16_bf16_flags() -> Tuple[bool, bool]:
    bf16 = is_bfloat16_supported()
    return (not bf16), bf16


def save_model_and_tokenizer(model, tokenizer, save_dir: str, original_base_model_path: Optional[str] = None):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Update adapter_config.json with original base model path if provided
    if original_base_model_path is not None:
        adapter_config_path = Path(save_dir) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            adapter_config["base_model_name_or_path"] = original_base_model_path
            with open(adapter_config_path, "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2, ensure_ascii=False)