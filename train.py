import argparse
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import model
from dataset import get_data_loaders
from utils.config_utils import load_config, normalize_config

torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


def build_transforms(img_size):
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
    
    root_folder = os.path.join(cfg['dataset_folder'], 'images')
    annotation_file = os.path.join(cfg['dataset_folder'], 'captions.txt')
    
    return get_data_loaders(
        root_dir=root_folder,
        captions_file=annotation_file,
        transform=transform,
        num_workers=cfg.get('num_workers', 4),
        batch_size=cfg.get('batch_size', 32),
    )


def setup_config(args):
    cfg = load_config(args.config)
    cfg = normalize_config(cfg)

    if args.override:
        for o in args.override:
            if '=' in o:
                k, v = o.split('=', 1)
                try:
                    cfg[k] = eval(v)
                except:
                    cfg[k] = v

    ckpt_path = args.ckpt_path or cfg.get('ckpt_path')
    if ckpt_path and args.use_ckpt_hparams:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            saved_hparams = checkpoint.get('hyper_parameters', {})
            for k, v in saved_hparams.items():
                if k in cfg:
                    cfg[k] = v
        except Exception as e:
            print(f"Warning: Failed to load hparams from {ckpt_path}: {e}")

    cfg['learning_rate'] = float(cfg.get('learning_rate', 1e-4))
    
    return cfg, ckpt_path


def main():
    parser = argparse.ArgumentParser(description='Train image captioning model')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to config YAML')
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'], help='Device to use')
    parser.add_argument('--ckpt_path', default=None, help='Resume training from this checkpoint')
    parser.add_argument('--use_ckpt_hparams', action='store_true', help='Use hparams from checkpoint')
    parser.add_argument('--override', nargs='*', help='Override config key=value pairs')
    args = parser.parse_args()
    
    cfg, ckpt_path = setup_config(args)
    print(f"Training Config: Batch Size={cfg['batch_size']}, LR={cfg['learning_rate']:.6f}")

    train_loader, val_loader, test_loader, dataset = build_dataloaders(cfg)

    caption_model = model.CaptionModel(
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        dataset=dataset,
        learning_rate=cfg['learning_rate']
    )
    
    if cfg.get('freeze_encoder', True):
        for param in caption_model.cnn.parameters():
            param.requires_grad = False

    logger = TensorBoardLogger(
        save_dir=cfg.get('tb_log_dir', 'tb_logs'), 
        name=cfg.get('logger_name', 'resnet_transformer')
    )
    
    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor='val_meteor', mode='max', filename='{epoch}-{val_meteor:.2f}'),
        EarlyStopping(monitor='val_meteor', patience=cfg.get('early_stop_patience', 6), mode='max', verbose=True),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        logger=logger,
        accelerator=args.device,
        devices=1,
        max_epochs=cfg['epochs'],
        precision=cfg.get('precision', '16-mixed') if args.device == 'gpu' else '32',
        callbacks=callbacks,
        log_every_n_steps=50
    )

    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")

    trainer.fit(caption_model, train_loader, val_loader, ckpt_path=ckpt_path)
    trainer.test(caption_model, test_loader)


if __name__ == '__main__':
    main()