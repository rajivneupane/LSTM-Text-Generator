#!/usr/bin/env python

import argparse
import torch
from src.model import LSTMLanguageModel
from src.data_handler import TextDataHandler

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained LSTM model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--prompt', type=str, default=None, help='Starting prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=200, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    args = parser.parse_args()

    # Load vocabulary
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data_handler = TextDataHandler(text)

    # Initialize and load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMLanguageModel(
        vocab_size=data_handler.vocab_size,
        device=device
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Generate text
    with torch.no_grad():
        if args.prompt:
            encoded_prompt = data_handler.encode(args.prompt)
            context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        generated_ids = model.generate(context, args.max_tokens)[0].tolist()
        
        if args.prompt:
            generated_text = args.prompt + data_handler.decode(generated_ids[len(encoded_prompt):])
        else:
            generated_text = data_handler.decode(generated_ids)
        
        print(generated_text)

if __name__ == '__main__':
    main()