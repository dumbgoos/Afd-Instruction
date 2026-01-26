import re
import torch
from typing import Dict, Any, List, Optional
from transformers import TextStreamer


class TextGenerator:
    """Text generation handler for model inference."""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup text streamer if enabled
        self.streamer = None
        if config.get("logging", {}).get("enable_streaming", True):
            self.streamer = TextStreamer(tokenizer)
    
    def prepare_inputs(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """
        Prepare input tokens from messages.
        
        Args:
            messages: List of message dictionaries with 'from' and 'value' keys
            
        Returns:
            Input tensor ready for generation
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        return inputs
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response from input messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated response text
        """
        inputs = self.prepare_inputs(messages)
        generation_cfg = self.config.get("generation", {})
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                streamer=self.streamer,
                max_new_tokens=generation_cfg.get("max_new_tokens", 2048),
                use_cache=generation_cfg.get("use_cache", True),
            )
        
        # Decode the full response
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract assistant response
        return self.extract_assistant_response(text)
    
    def extract_assistant_response(self, text: str) -> str:
        """
        Extract assistant response from full generated text.
        
        Args:
            text: Full generated text including prompt
            
        Returns:
            Extracted assistant response
        """
        # Look for assistant response in chat template format
        match = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, re.S)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for assistant response after the last human message
        match = re.search(r"<\|im_start\|>human.*?<\|im_end\|>\s*<\|im_start\|>assistant\s*(.*?)$", text, re.S)
        if match:
            return match.group(1).strip()
        
        # If no pattern matches, return the text as is
        return text.strip()
    
    def chat_single_turn(self, user_input: str) -> str:
        """
        Convenient method for single-turn chat.
        
        Args:
            user_input: User's input message
            
        Returns:
            Model's response
        """
        messages = [{"from": "human", "value": user_input}]
        return self.generate_response(messages)
