import torch
from typing import Dict, Any, Tuple
from transformers import logging
from unsloth import FastLanguageModel


def setup_logging(verbosity: str = "error"):
    """Set up logging configuration."""
    if verbosity == "error":
        logging.set_verbosity_error()
    elif verbosity == "warning":
        logging.set_verbosity_warning()
    elif verbosity == "info":
        logging.set_verbosity_info()
    else:
        logging.set_verbosity_error()


def load_model_for_inference(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load model and tokenizer for inference.
    
    Args:
        config: Configuration dictionary containing model settings
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_cfg = config["model"]
    
    # Setup logging
    logging_cfg = config.get("logging", {})
    setup_logging(logging_cfg.get("verbosity", "error"))
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_path"],
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )
    
    # Set model to inference mode
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer


