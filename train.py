import argparse
import os
import math
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import model
import pytorch_lightning as pl
from dataset import get_data_loaders
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.config_utils import load_config, normalize_config

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


def build_transforms(img_size):
    """Return augmentation pipeline for training images."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def build_dataloaders(cfg):
    transform = build_transforms(cfg['img_size'])
    train_loader, val_loader, test_loader, dataset = get_data_loaders(
        root_folder=os.path.join(cfg['dataset_folder'], 'images'),
        annotation_file=os.path.join(cfg['dataset_folder'], 'captions.txt'),
        transform=transform,
        num_workers=cfg['num_workers'],
        batch_size=cfg['batch_size'],
        split_ratios=cfg['split_ratios']
    )
    return train_loader, val_loader, test_loader, dataset

def build_model(cfg, dataset):
    """Return an initialized `CaptionModel` instance using `cfg` and `dataset`."""
    caption_model = model.CaptionModel(
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        dataset=dataset,
        dataset_folder=cfg['dataset_folder'],
        learning_rate=cfg['learning_rate']
    )
    # Optionally freeze encoder
    if cfg.get('freeze_encoder', True):
        for param in caption_model.cnn.parameters():
            param.requires_grad = False
    return caption_model


def main():
    parser = argparse.ArgumentParser(description='Train image captioning model')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config YAML')
    parser.add_argument('--device', default='gpu', help='Device: gpu or cpu')
    parser.add_argument('--override', nargs='*', help='Override config key=value pairs')
    parser.add_argument('--ckpt_path', default=None, help='Path to a checkpoint to resume training (optional)')
    parser.add_argument('--use_ckpt_hparams', action='store_true', help='If a checkpoint is provided, use its hyperparameters to initialize training')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = normalize_config(cfg)
    if args.override:
        for o in args.override:
            if '=' not in o: continue
            k, v = o.split('=', 1)
            try:
                v_cast = eval(v)
            except Exception:
                v_cast = v
            cfg[k] = v_cast
        cfg = normalize_config(cfg)

    ckpt = args.ckpt_path or cfg.get('ckpt_path')
    ckpt_hparams = {}
    if ckpt:
        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            ckpt_hparams = checkpoint.get('hyper_parameters', checkpoint.get('hparams', {})) or {}
        except Exception as e:
            print(f"Failed to load checkpoint at {ckpt}: {e}")
            ckpt_hparams = {}

    use_ckpt = args.use_ckpt_hparams or bool(ckpt)
    runtime_cfg = dict(cfg)
    if ckpt and use_ckpt and ckpt_hparams:
        overridden = []
        for k, v in ckpt_hparams.items():
            if k in runtime_cfg:
                runtime_cfg[k] = v
                overridden.append(k)
        runtime_cfg = normalize_config(runtime_cfg)

    try:
        base_lr = float(runtime_cfg.get('base_lr', 1e-4))
        batch_size = int(runtime_cfg.get('batch_size', 128))
        base_batch = int(runtime_cfg.get('base_batch', 64))
        runtime_cfg['learning_rate'] = base_lr * math.sqrt(batch_size / base_batch)
    except Exception as e:
        print('Error computing scaled learning rate. Config types:')
        for k in ['base_lr', 'batch_size', 'base_batch']:
            print(k, type(runtime_cfg.get(k)), runtime_cfg.get(k))
        raise e
    print(f"Starting training with batch size={runtime_cfg['batch_size']}, lr={runtime_cfg['learning_rate']:.6f}")

    logger = TensorBoardLogger(runtime_cfg.get('tb_log_dir', 'tb_logs'), name=runtime_cfg.get('logger_name', 'resnet_transformer'))
    checkpoint_cb = ModelCheckpoint(save_top_k=1, monitor='val_bleu', mode='max')
    desired_precision = runtime_cfg.get('precision', '16-mixed')
    if args.device != 'gpu':
        desired_precision = '32'

    # Use a single device by default 
    trainer = pl.Trainer(
        logger=logger,
        accelerator=args.device,
        devices=1,
        max_epochs=runtime_cfg['epochs'],
        precision=desired_precision,
        callbacks=[
            EarlyStopping(monitor='val_bleu', patience=runtime_cfg.get('early_stop_patience', 6), verbose=True, mode='max'),
            LearningRateMonitor(logging_interval='epoch'),
            checkpoint_cb
        ]
    )

    ckpt = args.ckpt_path or cfg.get('ckpt_path')
    if ckpt:
        print(f"Resuming training from checkpoint: {ckpt}")
        
    train_loader, val_loader, test_loader, dataset = build_dataloaders(runtime_cfg)
    caption_model = build_model(runtime_cfg, dataset)
    trainer.fit(caption_model, train_loader, val_loader, ckpt_path=ckpt)
    trainer.test(caption_model, test_loader)


if __name__ == '__main__':
    main()