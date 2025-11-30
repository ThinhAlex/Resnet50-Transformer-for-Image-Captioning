# Image Captioning on Flickr8k

This repository implements an image captioning model using a ResNet encoder
and a Transformer decoder on the Flickr8k dataset. The project uses
PyTorch Lightning for training, timm for the encoder, and TorchMetrics for
BLEU evaluation.

## Features
- ResNet-based CNN encoder (timm)
- Transformer-based decoder
- Tokenizer and vocabulary using spaCy
- Configuration via YAML and CLI flags
- Checkpointing with hyperparameters
- Automated inference which recovers configuration from the saved checkpoint

## Quick Start
1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Prepare dataset: ensure `flickr8k/images` and `flickr8k/captions.txt` exist
	 and that `captions.txt` contains image filenames and captions.

3. Train with the default config:
```powershell
python train.py --config configs/default.yaml
```

4. Run inference using a checkpoint:
```powershell
python inference_save.py --checkpoint tb_logs/resnet_transformer/version_*/checkpoints/epoch=*.ckpt --img_size=224
```

The inference script exports example images to `inference_results/` with predicted
and real captions shown under the image. Example output thumbnails are in
the repo's `inference_results/` folder (if any). If you want to inspect results
manually, open the saved PNG images or use Jupyter to display them in a cell.

### Resume training from a checkpoint
To resume training from checkpoint, you can either provides CLI argument `--ckpt_path` to `train.py` OR pass a checkpoint path to `ckpt_path` in YAML file.

Example (CLI):
```powershell
python train.py --config configs/default.yaml --ckpt_path tb_logs/resnet_transformer/version_5/checkpoints/epoch=22-step=5819.ckpt
```

Example (YAML config): add `ckpt_path: tb_logs/resnet_transformer/version_5/checkpoints/epoch=22-step=5819.ckpt` to `configs/default.yaml`.

If `ckpt_path` is not set, training will start from scratch.

Note:
- If YAML configs do not match the checkpoint, training would resume with the checkpoint hyperparameters. 
- If you forget the checkpoint hyperparameters, the project also includes `scripts/inspect_ckpt.py` which prints saved hparams from a checkpoint.


## Configuration
All training and inference parameters are stored in `configs/default.yaml`. You
can override values on the command line with `--override key=value` pairs, e.g.

```powershell
python train.py --config configs/default.yaml --override batch_size=64 hidden_size=256
```

Note: When passing values with `--override`, use Python literals (e.g. `batch_size=64`, `base_lr=1e-4`). Using strings or lists can cause type errors (e.g., when computing scaled learning rate). If you see a TypeError about multiplying a sequence by a float, check the config values types.

## Files
- `dataset.py`: dataset and vocabulary implementation
- `model.py`: model (encoder + decoder), training routines, inference helpers
- `train.py`: training script (YAML/CLI-driven)
- `inference_save.py`: generate images + predicted captions saved to `inference_results/`
- `tb_logs/`: tensorboard logs & model checkpoints (created during training)
- `configs/default.yaml`: default training configs
- `requirements.txt`: pip install requirements

### Example Project Layout
```
.
├── configs/
│   └── default.yaml
├── flickr8k/
│   ├── captions.txt
│   └── images/
├── tb_logs/
│   └── resnet_transformer/
│       └── version_x/
│           └── checkpoints/
├── inference_results/
├── train.py
├── inference_save.py
├── model.py
├── dataset.py
├── README.md
└── requirements.txt
```

## Troubleshooting
- If your checkpoint can't be loaded, verify the model signature in `model.py`
	and ensure `dataset_folder` or `dataset` is provided or recoverable from the
	checkpoint. Also update packages to match `requirements.txt`.

