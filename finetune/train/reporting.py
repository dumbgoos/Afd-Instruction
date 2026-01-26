import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import neptune
except Exception:
    neptune = None


def init_neptune(cfg: Dict[str, Any]):
    if neptune is None:
        print("[reporting] Neptune is not installed. Skipping.")
        return None
    proj = cfg.get("reporting", {}).get("project")
    tags = cfg.get("reporting", {}).get("tags", [])
    mode = cfg.get("reporting", {}).get("mode", "async")

    run = neptune.init_run(project=proj, mode=mode)
    if tags:
        run["sys/tags"].add(tags)

    run["params/max_seq_length"] = cfg.get("max_seq_length", None)
    run["params/seed"] = cfg.get("seed", None)
    if "unsloth" in cfg:
        run["unsloth"] = cfg["unsloth"]
    if "lora" in cfg:
        run["lora"] = cfg["lora"]
    if "training" in cfg:
        run["training"] = cfg["training"]

    return run


def log_config_artifact(run, config_path: Path):
    if run is None:
        return
    run["artifacts/config"].upload(str(config_path))


def finalize_neptune(run):
    if run is None:
        return
    run.stop()