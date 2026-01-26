import os
import json
import argparse
from pathlib import Path

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from training_loop import build_trainer_and_tokenizer
from model_builder import save_model_and_tokenizer
from reporting import init_neptune, finalize_neptune, log_config_artifact


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.cuda_visible_devices is not None:
        cfg.setdefault("cuda_visible_devices", args.cuda_visible_devices)
    if args.data_file is not None:
        cfg.setdefault("paths", {}).update({"data_file": args.data_file})
    if args.base_model is not None:
        cfg.setdefault("paths", {}).update({"base_model": args.base_model})
    if args.output_dir is not None:
        cfg.setdefault("paths", {}).update({"output_dir": args.output_dir})
    if args.save_dir is not None:
        cfg.setdefault("paths", {}).update({"save_dir": args.save_dir})
    if args.learning_rate is not None:
        cfg.setdefault("training", {}).update({"learning_rate": args.learning_rate})
    if args.num_train_epochs is not None:
        cfg.setdefault("training", {}).update({"num_train_epochs": args.num_train_epochs})
    if args.per_device_train_batch_size is not None:
        cfg.setdefault("training", {}).update({"per_device_train_batch_size": args.per_device_train_batch_size})
    if args.max_seq_length is not None:
        cfg["max_seq_length"] = args.max_seq_length
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning training script")
    p.add_argument("--config", type=str, required=True,
                   help="Path to JSON config (e.g., ../config/train_param.json)")
    p.add_argument("--cuda_visible_devices", type=str)
    p.add_argument("--data_file", type=str)
    p.add_argument("--base_model", type=str)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--save_dir", type=str)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--num_train_epochs", type=int)
    p.add_argument("--per_device_train_batch_size", type=int)
    p.add_argument("--max_seq_length", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = merge_overrides(load_config(cfg_path), args)

    if cfg.get("cuda_visible_devices") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])

    run = None
    if cfg.get("reporting", {}).get("use_neptune", False):
        run = init_neptune(cfg)
        if cfg.get("reporting", {}).get("upload_config", True):
            log_config_artifact(run, cfg_path)

    trainer, tokenizer, save_dir = build_trainer_and_tokenizer(cfg, neptune_run=run)
    
    trainer.train()

    # Get original base model path from config
    original_base_model_path = cfg.get("paths", {}).get("base_model")
    save_model_and_tokenizer(trainer.model, tokenizer, save_dir, original_base_model_path=original_base_model_path)

    finalize_neptune(run)
    print("Training completed!")


if __name__ == "__main__":
    main()
