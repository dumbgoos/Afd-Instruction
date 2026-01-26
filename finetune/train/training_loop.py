from typing import Dict, Any, Tuple
from transformers import TrainingArguments
from trl import SFTTrainer

from data_io import prepare_tokenizer_with_template, load_and_prepare_dataset
from model_builder import build_model_and_tokenizer, fp16_bf16_flags
from transformers.integrations import NeptuneCallback


def build_trainer_and_tokenizer(cfg: Dict[str, Any], neptune_run=None) -> Tuple[SFTTrainer, Any, str]:
    paths = cfg["paths"]
    max_seq_length = int(cfg.get("max_seq_length", 2048))

    model, tokenizer = build_model_and_tokenizer(
        paths=paths,
        max_seq_length=max_seq_length,
        chat_template_cfg=cfg.get("chat_template", {}),
        unsloth_cfg=cfg.get("unsloth", {}),
        lora_cfg=cfg.get("lora", {}),
    )

    tokenizer = prepare_tokenizer_with_template(tokenizer, cfg.get("chat_template", {}))

    dataset = load_and_prepare_dataset(paths["data_file"], tokenizer)

    fp16_flag, bf16_flag = fp16_bf16_flags()
    t = cfg.get("training", {})
    training_args = TrainingArguments(
        learning_rate=float(t.get("learning_rate", 3e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 8)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 2)),
        num_train_epochs=int(t.get("num_train_epochs", 1)),
        fp16=fp16_flag,
        bf16=bf16_flag,
        logging_steps=int(t.get("logging_steps", 1)),
        optim=t.get("optim", "adamw_8bit"),
        weight_decay=float(t.get("weight_decay", 0.01)),
        warmup_steps=int(t.get("warmup_steps", 10)),
        output_dir=paths.get("output_dir", "output"),
        seed=int(cfg.get("seed", 0)),
        report_to=["neptune"] if cfg.get("reporting", {}).get("use_neptune", False) else ["none"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    if cfg.get("reporting", {}).get("use_neptune", False):
        trainer.add_callback(NeptuneCallback(run=neptune_run) if neptune_run is not None else NeptuneCallback())

    return trainer, tokenizer, paths.get("save_dir", "model")