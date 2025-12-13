import argparse
import os
import torch
import textwrap
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import model
from dataset import get_data_loaders

try:
    from utils.config_utils import load_config, normalize_config
except ImportError:
    def load_config(path):
        return {}
    def normalize_config(cfg):
        return cfg

def main():
    parser = argparse.ArgumentParser(description='Run inference and save results')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--results', default='inference_results', help='Folder to save results')
    parser.add_argument('--num_examples', type=int, default=20, help='Number of examples to generate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size override')
    parser.add_argument('--dataset_folder', type=str, default='flickr8k', help='Root folder of dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results, exist_ok=True)

    # Validation transforms
    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ])

    # Load dataset and test loader
    print(f"Loading dataset from {args.dataset_folder}...")
    _, _, test_loader, dataset = get_data_loaders(
        root_dir=os.path.join(args.dataset_folder, 'images'),
        captions_file=os.path.join(args.dataset_folder, 'captions.txt'),
        transform=transform,
        batch_size=1,
        num_workers=0,
    )
    
    print(f"Loading model from {args.checkpoint}...")
    
    try:
        loaded_model = model.CaptionModel.load_from_checkpoint(
            args.checkpoint, 
            dataset=dataset
        )
    except Exception as e:
        print(f"Warning: Direct load failed ({e}). Attempting to load with strict=False...")
        loaded_model = model.CaptionModel.load_from_checkpoint(
            args.checkpoint, 
            dataset=dataset,
            strict=False
        )

    loaded_model.to(device)
    loaded_model.eval()

    print(f"Generating {args.num_examples} examples...")

    # Inference loop
    for idx, batch in enumerate(test_loader):
        if idx >= args.num_examples:
            break
            
        img, _, img_ids = batch
        img = img.to(device)
        
        # generate_caption expects (C,H,W); squeeze batch dim
        gen_text = loaded_model.generate_caption(img.squeeze(0))
        
        img_id = img_ids[0]
        refs = dataset.df[dataset.df['image'] == img_id]['caption'].tolist()
        
        # --- CHANGED HERE: Take only the first reference caption ---
        if refs:
            real_text = refs[0]
        else:
            real_text = "<No Reference>"
        # -----------------------------------------------------------
        
        gen_disp = "\n".join(textwrap.wrap(gen_text, width=50))
        real_disp = "\n".join(textwrap.wrap(real_text, width=50))

        # Denormalize for display
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        disp_img = img.squeeze(0) * std + mean
        disp_img = torch.clamp(disp_img, 0, 1).permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp_img)
        ax.axis("off")
        
        caption_text = f"REAL:\n{real_disp}\n\nPRED:\n{gen_disp}"
        
        ax.text(
            0.5, -0.05, caption_text, transform=ax.transAxes, 
            horizontalalignment='center', verticalalignment='top', 
            fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        plt.subplots_adjust(bottom=0.3)
        save_path = os.path.join(args.results, f"result_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()