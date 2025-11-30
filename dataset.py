import os
from typing import Tuple, Optional

import pandas as pd
import spacy
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Ensure spaCy model is downloaded with: python -m spacy download en_core_web_sm
try:
    spacy_eng = spacy.load("en_core_web_sm")
except:
    print("Downloading spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, lower_threshold, upper_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        for word in frequencies:
            if (frequencies[word] >= self.lower_threshold) and (frequencies[word] <= self.upper_threshold):
                if word not in self.stoi:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    """Simple PyTorch Dataset for Flickr8k-style captioning datasets.

    Args:
        root_dir: Path to folder with images
        captions_file: CSV or tab file with columns `image` and `caption`
        transform: albumentations transform pipeline
        lower_threshold: Min frequency for words to be included in vocabulary
        upper_threshold: Max frequency for words to be included in vocabulary
    """
    def __init__(self, root_dir: str, captions_file: str, transform=None, lower_threshold=5, upper_threshold=500000):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.vocab = Vocabulary(lower_threshold, upper_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        img = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

def get_data_loaders(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    split_ratios=(0.8, 0.1, 0.1)
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    train_size = int(split_ratios[0] * len(dataset))
    val_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    collate_fn = MyCollate(pad_idx=pad_idx)
    
    # Use persistent_workers if num_workers > 0 (speeds up on Windows)
    use_persistent = True if num_workers > 0 else False

    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, 
        shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn, persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=num_workers, 
        shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn, persistent_workers=use_persistent
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, 
        shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn, persistent_workers=use_persistent
    )

    return train_loader, val_loader, test_loader, dataset