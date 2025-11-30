import argparse
import torch
import json


def main():
    parser = argparse.ArgumentParser(description='Print hyperparameters from a PyTorch Lightning checkpoint')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', ckpt.get('hparams', {})) or {}
    metadata = {
        'checkpoint_path': args.checkpoint,
        'hparams': hparams,
    }
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
