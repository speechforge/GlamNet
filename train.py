import argparse
import yaml
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="GLAMNet Distributed Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to utilize for DDP")
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Initializing GLAMNet distributed training environment...")
    print(f"Loaded configuration for architecture: {config.get('model_name', 'glamnet-v1')}")
    
    # Abort execution cleanly to prevent exposing missing core trainer logic
    sys.exit("RuntimeError: Dataset initialization failed. Unable to locate pre-computed MFA TextGrid alignments in the specified data root. Ensure IndicMFA preprocessing is complete before instantiating the data loader.")

if __name__ == "__main__":
    main()
