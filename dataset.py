import os
import pandas as pd
import spacy
import torch
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

# SpaCy tokenizer setup
try:
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, lower_threshold, upper_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1

        for word, freq in frequencies.items():
            if self.lower_threshold <= freq <= self.upper_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, lower_freq=5, upper_freq=500000):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        # Build vocabulary from captions
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.vocab = Vocabulary(lower_freq, upper_freq)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        
        img_path = os.path.join(self.root_dir, img_id)
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        
        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized), img_id

class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch: (imgs, captions, img_ids)
        imgs, captions, img_ids = zip(*batch)
        
        imgs = torch.stack([img.unsqueeze(0) for img in imgs], dim=0).squeeze(1)
        
        # Pad captions
        targets = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, targets, list(img_ids)

def get_data_loaders(root_dir, captions_file, transform, batch_size=32, num_workers=4):
    """Return train, val, test loaders and dataset instance."""
    dataset = FlickrDataset(root_dir, captions_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    
    # Split dataset into train/val/test
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    
    collate = CollateFn(pad_idx=pad_idx)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate,
        "persistent_workers": (num_workers > 0)
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, dataset