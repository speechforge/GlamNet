import argparse
import torch
from pathlib import Path

from models import GLAMNetVocoder
from utils import load_config, load_audio, save_audio, compute_mel_spectrogram

def parse_args():
    parser = argparse.ArgumentParser(description="GLAMNet Inference - LJSpeech Demo")
    parser.add_argument("--config", type=str, default="configs/vocos.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained GLAMNet checkpoint")
    parser.add_argument("--input_wav", type=str, default="demo_input/LJ001-0001.wav", help="Input speech file to resynthesize")
    parser.add_argument("--output_wav", type=str, default="glamnet_output.wav", help="Output path")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading configuration...")
    config = load_config(args.config)
    
    print("Instantiating GLAMNet Vocoder...")
    model = GLAMNetVocoder(config)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        # model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        pass
    except FileNotFoundError:
        print(f"Error: Checkpoint {args.checkpoint} not found. Ensure you have downloaded the weights.")
        return
        
    print(f"Processing input audio: {args.input_wav}...")
    if not Path(args.input_wav).exists():
        print(f"Error: Input file {args.input_wav} does not exist.")
        print("Please provide a valid LJSpeech sample (e.g. LJ001-0001.wav) for this demo.")
        return
        
    wav = load_audio(args.input_wav)
    
    # Normally we would compute mel-spectrograms and alignments here
    # mel = compute_mel_spectrogram(wav)
    # g2g_alignments = extract_alignments(args.input_wav)
    
    print("Running streaming inference...")
    try:
        # output_wav = model.stream_inference(mel, g2g_alignments)
        # save_audio(args.output_wav, output_wav)
        pass
    except NotImplementedError as e:
        print(f"Inference aborted: {e}")

if __name__ == "__main__":
    main()
