import unsloth
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from model_loader import load_model_for_inference
from generator import TextGenerator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command line arguments into configuration."""
    if args.model_path:
        config["model"]["model_path"] = args.model_path
    if args.max_new_tokens:
        config["generation"]["max_new_tokens"] = args.max_new_tokens
    
    return config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument(
        "--config", 
        type=str, 
        default="../config/infer_param.json",
        help="Path to inference configuration file"
    )
    parser.add_argument("--model_path", type=str, help="Override model path")
    parser.add_argument("--max_new_tokens", type=int, help="Maximum new tokens to generate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    return parser.parse_args()


def run_example_inference(generator: TextGenerator):
    """Run the example inference from the original code."""
    print("Running example inference...")
    print("=" * 50)
    
    # Example 1: Original antibody sequence question
    user_input1 = ("<H>EVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSA</H>, "
                  "<L>YELTQETGVSVALGDTVTITCEGDSLESHYASWYQKKPGQAPILLFYGDDNRPSGVPDRFSGDADGNEASLTIDGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQ</L> "
                  "Are the mutations on the light chain surface responsible for increasing neutralization potency?")
    
    print("Question 1:")
    print(user_input1)
    print("\nAnswer 1:")
    response1 = generator.chat_single_turn(user_input1)
    print(response1)
    print("\n" + "=" * 50)
    
    # Example 2: Dimerization question
    user_input2 = ("<H>EVQLQQSGPELVKPGASVKISCKASGYSFTGYFMNWVKQSHGKSLEWIGRINPYNGDTFYKQRFKGKATLTVDKSSSTVHMDLLSLTSEDSAVYYCGRGNYYFDYWGQGTTLTVSSA</H> "
                  "<L>DVQITQSPSYLAASPGETITINCRTSKNISKYLAWYQEKPGKTNKLLIYSGYTLQSGIPSRFSGSGSGTDFTLTISSLEPEDFAMYYCQQHNEYPYTFGGGTKLEIKRA</L> "
                  "What type of dimerization does this antibody promote?")
    
    print("Question 2:")
    print(user_input2)
    print("\nAnswer 2:")
    response2 = generator.chat_single_turn(user_input2)
    print(response2)
    print("\n" + "=" * 50)


def run_interactive_mode(generator: TextGenerator):
    """Run interactive chat mode."""
    print("Interactive mode started. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
                
            print("Assistant: ", end="")
            response = generator.chat_single_turn(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load and merge configuration
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)
    
    print(f"Loading model from: {config['model']['model_path']}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(config)
    
    # Create generator
    generator = TextGenerator(model, tokenizer, config)
    
    print("Model loaded successfully!")
    
    if args.interactive:
        run_interactive_mode(generator)
    else:
        run_example_inference(generator)


if __name__ == "__main__":
    main()
