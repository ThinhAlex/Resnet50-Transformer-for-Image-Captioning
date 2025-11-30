import argparse
import os
import torch
import yaml
from utils.config_utils import load_config, normalize_config
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import model
from dataset import get_data_loaders
import textwrap


def main():
    parser = argparse.ArgumentParser(description='Run inference and save results')
    parser.add_argument('--config', default='configs/default.yaml', help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--results', default='inference_results', help='Folder to save results')
    parser.add_argument('--num_examples', type=int, default=20, help='Number of examples to generate')
    parser.add_argument('--img_size', type=int, help='Image size override')
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        try:
            cfg = load_config(args.config)
        except Exception:
            cfg = {}
    cfg = normalize_config(cfg)

    CHECKPOINT_PATH = args.checkpoint
    RESULT_FOLDER = args.results
    NUM_EXAMPLES = args.num_examples
    IMG_SIZE = args.img_size or cfg.get('img_size', 224)

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    saved_hparams = checkpoint.get('hyper_parameters', checkpoint.get('hparams', {})) or {}
    runtime_cfg = dict(cfg)
    for k, v in saved_hparams.items():
        if k in runtime_cfg:
            runtime_cfg[k] = v
    runtime_cfg = normalize_config(runtime_cfg)
    dataset_folder = runtime_cfg.get('dataset_folder', cfg.get('dataset_folder', 'flickr8k'))
    IMG_SIZE = args.img_size or runtime_cfg.get('img_size', IMG_SIZE)
    batch_size = 1
    _, _, test_loader, dataset = get_data_loaders(
        root_folder=os.path.join(dataset_folder, 'images'),
        annotation_file=os.path.join(dataset_folder, 'captions.txt'),
        transform=transform,
        batch_size=batch_size,
        num_workers=0
    )

    print("Loading model from checkpoint...")
    try:
        loaded_model = model.CaptionModel.load_from_checkpoint(CHECKPOINT_PATH, dataset=dataset)
    except TypeError:
        init_args = {k: saved_hparams[k] for k in ('hidden_size', 'num_layers', 'learning_rate') if k in saved_hparams}
        loaded_model = model.CaptionModel.load_from_checkpoint(CHECKPOINT_PATH, dataset=dataset, **init_args)
    loaded_model.to(device)
    loaded_model.eval()
    if getattr(loaded_model, 'corpus', None) is None:
        loaded_model.set_dataset(dataset)

    print("Generating inference results...")

    for idx, (img, label) in enumerate(test_loader):
        if idx >= NUM_EXAMPLES:
            break
            
        gen_caption_list = loaded_model.generate_caption_example(img.squeeze(0))
        gen_text = " ".join(gen_caption_list)
        
        real_caption = [dataset.vocab.itos[i.item()] for i in label[0] if i.item() not in [0,1,2]]
        real_text = " ".join(real_caption)
        
        gen_text = "\n".join(textwrap.wrap(gen_text, width=60))
        real_text = "\n".join(textwrap.wrap(real_text, width=60))

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        disp_img = img.squeeze(0).cpu() * std + mean
        disp_img = torch.clamp(disp_img, 0, 1).permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(disp_img)
        ax.axis("off")
        
        caption_text = r"$\bf{REAL}$: " + real_text + "\n" + r"$\bf{PRED}$: " + gen_text
        
        ax.text(
            0.5, -0.02, caption_text, transform=ax.transAxes, 
            horizontalalignment='center', verticalalignment='top', 
            fontsize=12, color='black'
        )

        plt.subplots_adjust(bottom=0.15)
        save_path = os.path.join(RESULT_FOLDER, f"result_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"Saved {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()