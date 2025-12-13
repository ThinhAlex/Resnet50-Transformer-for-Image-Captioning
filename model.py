import math
import os
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.text import BLEUScore

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    
    # It might take a while to download the first time
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
except ImportError:
    meteor_score = None
    word_tokenize = None
    
import warnings
warnings.filterwarnings("ignore", "Support for mismatched key_padding_mask")
warnings.filterwarnings("ignore", "Trying to infer the `batch_size`")    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CaptionModel(pl.LightningModule):
    def __init__(self, hidden_size, num_layers, dataset, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['dataset'])
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.learning_rate = learning_rate
        self.d_model = hidden_size
        self.pad_idx = self.vocab.stoi["<PAD>"]
        
        # Architecture: ResNet50 (Encoder) -> Projection Layer -> Transformer (Decoder)
        self.cnn = timm.create_model('resnet50', pretrained=True)
        self.cnn.fc = nn.Identity()
        self.cnn_proj = nn.Linear(self.cnn.num_features, self.d_model)
        self.cnn_dropout = nn.Dropout(0.1)
        
        self.embedding = nn.Embedding(len(self.vocab), self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=4*hidden_size, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.d_model, len(self.vocab))
        
        # Loss and val_ref mapping
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=0.1)

        if hasattr(dataset, 'df'):
            self.val_refs = dataset.df.groupby('image')['caption'].apply(list).to_dict()
        else:
            self.val_refs = {}

        # BLEU-1 -> BLEU-5 metrics 
        self.bleu_metrics = nn.ModuleDict({f'bleu{k}': BLEUScore(n_gram=k, smooth=True) for k in range(1, 6)})

    def forward(self, images, captions):
        features = self.cnn.forward_features(images) 
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1) 
        memory = self.cnn_dropout(self.pos_encoder(self.cnn_proj(features)))
        
        tgt = captions[:, :-1]
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        tgt_mask = self.generate_mask(tgt.size(1)).to(images.device)
        tgt_pad_mask = (tgt == self.pad_idx)

        output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        return self.classifier(output)

    def generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        if batch_idx % 200 == 0:
            self.log_tb_images(batch[0][0], batch[1][0], batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        if batch_idx % 50 == 0:
            images, _, img_ids = batch
            metrics = self.compute_metrics(images, img_ids)
            for k, v in metrics.items():
                self.log(f"val_{k}", v, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        
        images, _, img_ids = batch
        metrics = self.compute_metrics(images, img_ids)
        for k, v in metrics.items():
            self.log(f"test_{k}", v, on_epoch=True, prog_bar=True)
        return loss

    def _compute_loss(self, batch):
        images, captions, _ = batch
        outputs = self.forward(images, captions)
        targets = captions[:, 1:] 
        return self.loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

    def compute_metrics(self, images, img_ids):
        preds, refs = [], []
        for i, image in enumerate(images):
            preds.append(self.generate_caption(image))
            
            img_id = img_ids[i].decode('utf-8') if isinstance(img_ids[i], bytes) else str(img_ids[i])
            if img_id in self.val_refs:
                refs.append(self.val_refs[img_id])
            else:
                refs.append(self.val_refs.get(os.path.basename(img_id), [""]))

        metrics = {}
        for name, metric in self.bleu_metrics.items():
            metrics[name] = float(metric(preds, refs))
            try: 
                metric.reset()
            except: 
                pass

        meteor_vals = []
        if meteor_score is not None:
            for p, r_list in zip(preds, refs):
                meteor_vals.append(meteor_score([word_tokenize(r) for r in r_list], word_tokenize(p)))
        
        metrics['meteor'] = sum(meteor_vals) / len(meteor_vals) if meteor_vals else 0.0
        return metrics

    def generate_caption(self, image, max_length=20):
        was_training = self.training 
        self.eval() 
        
        with torch.no_grad():
            img = image.unsqueeze(0).to(self.device)
            features = self.cnn.forward_features(img)
            B, C, H, W = features.shape
            features = features.view(B, C, H * W).permute(0, 2, 1)
            memory = self.cnn_dropout(self.pos_encoder(self.cnn_proj(features)))

            generated = torch.tensor([[self.vocab.stoi["<SOS>"]]], device=self.device)
            for _ in range(max_length):
                tgt_emb = self.pos_encoder(self.embedding(generated))
                tgt_mask = self.generate_mask(generated.size(1)).to(self.device)
                out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                next_token = self.classifier(out[:, -1, :]).argmax(1).item()
                
                if next_token == self.vocab.stoi["<EOS>"]: 
                    break
                
                generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=1)

        tokens = [self.vocab.itos[idx.item()] for idx in generated[0] if idx.item() not in [0, 1]]
        
        # Restore training state
        if was_training: 
            self.train()
        return " ".join(tokens)

    def log_tb_images(self, img, lbl, step):
        if self.logger:
            pred = self.generate_caption(img)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3,1,1)
            disp_img = torch.clamp(img * std + mean, 0, 1)
            self.logger.experiment.add_image('Images', disp_img, step)
            self.logger.experiment.add_text('Captions', f"Pred: {pred}", step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_bleu"}}