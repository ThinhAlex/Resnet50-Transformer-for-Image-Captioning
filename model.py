import torch
import torch.nn as nn
import timm
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics.text import BLEUScore  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CaptionModel(pl.LightningModule):
    """Image captioning model using a CNN encoder (ResNet) and a Transformer decoder.

    Only `hidden_size` is used as the model embedding dimension (d_model).
    The `dataset` is optional; if provided the vocabulary size will be taken
    from dataset.vocab. If not provided, the embedding layer is created lazily
    or with a placeholder and re-initialized when `dataset` is set.
    """
    def __init__(self, hidden_size, num_layers, dataset=None, learning_rate=1e-4, dataset_folder: str = None, **kwargs):
        super(CaptionModel, self).__init__()
        # Save hyperparameters 
        self.save_hyperparameters(ignore=['dataset'])
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.corpus = dataset.vocab if dataset is not None else None
        
        # Transformer parameters
        self.d_model = hidden_size
        self.num_heads = 8
        self.dim_feedforward = 4 * hidden_size
        self.dropout_p = 0.1
        
        # CNN encoder 
        self.cnn = timm.create_model('resnet50', pretrained=True)
        self.cnn.fc = nn.Identity() 
        
        # Linear projection for CNN features -> d_model
        self.input_channels = self.cnn.num_features
        self.cnn_proj = nn.Linear(self.input_channels, self.d_model)
        self.cnn_dropout = nn.Dropout(self.dropout_p)
        
        if dataset is not None:
            num_embeddings = len(dataset.vocab)
        else:
            num_embeddings = 1
        
        # Transformer decoder
        self.embedding = nn.Embedding(num_embeddings, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, 
            nhead=self.num_heads, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout_p,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.d_model, len(dataset.vocab))
        
        # Loss and metrics
        pad_index = dataset.vocab.stoi["<PAD>"] if dataset is not None else 0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index, label_smoothing=0.1)
        self.bleu_score = BLEUScore(n_gram=4, smooth=True)

    def forward(self, images, captions):
        features = self.cnn.forward_features(images) 
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1) 
        
        memory = self.cnn_proj(features)
        memory = self.pos_encoder(memory)
        memory = self.cnn_dropout(memory)
        
        tgt = captions[:, :-1] 
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(images.device)
        tgt_padding_mask = (tgt == self.corpus.stoi["<PAD>"])

        output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return self.classifier(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
        
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        images, labels = batch
        outputs = self.forward(images, labels)
        
        targets = labels[:, 1:]
        loss = self.loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
        
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        
        if stage != "train" or batch_idx % 100 == 0:
             self.log(f"{stage}_bleu", self.compute_bleu_score(outputs.detach(), targets), on_epoch=True, prog_bar=True)
        
        if batch_idx % 200 == 0 and stage == "train":
            self.log_tb_images(images[0], labels[0], batch_idx)
            
        return loss

    def log_tb_images(self, image_tensor, label_tensor, step):
        if self.logger:
            pred_caption = self.generate_caption_example(image_tensor)
            pred_text = ' '.join(pred_caption)
            real_caption = [self.corpus.itos[idx.item()] for idx in label_tensor if idx.item() not in [0,1,2]]
            real_text = ' '.join(real_caption)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).to(image_tensor.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(image_tensor.device).view(3, 1, 1)
            disp_img = torch.clamp(image_tensor * std + mean, 0, 1)
            
            self.logger.experiment.add_image('Generated_Images', disp_img, step)
            self.logger.experiment.add_text('Captions', f"**Real:** {real_text}  \n**Pred:** {pred_text}", step)

    def generate_caption_example(self, image, max_length=20):
        self.eval()
        image = image.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            features = self.cnn.forward_features(image)
            B, C, H, W = features.shape
            features = features.view(B, C, H * W).permute(0, 2, 1)
            memory = self.cnn_proj(features)
            memory = self.pos_encoder(memory)
            
            start_token = self.corpus.stoi["<SOS>"]
            generated = torch.tensor([[start_token]], device=self.device)
            
            for _ in range(max_length):
                tgt_emb = self.embedding(generated)
                tgt_emb = self.pos_encoder(tgt_emb)
                
                seq_len = generated.size(1)
                tgt_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)
                
                output = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                last_output = output[:, -1, :] 
                logits = self.classifier(last_output)
                
                next_token = logits.argmax(1).item()
                generated = torch.cat([generated, torch.tensor([[next_token]], device=self.device)], dim=1)
                
                if next_token == self.corpus.stoi["<EOS>"]:
                    break
        
        result_words = [self.corpus.itos[idx.item()] for idx in generated[0] if idx.item() not in [0, 1]]
        return result_words

    def set_dataset(self, dataset):
        """Attach a dataset and reinitialize vocab-dependent modules."""
        self.dataset = dataset
        self.corpus = dataset.vocab
        # Recreate embedding and loss sized to vocabulary
        self.embedding = nn.Embedding(len(self.corpus), self.d_model)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.corpus.stoi["<PAD>"], label_smoothing=0.1)

    def compute_bleu_score(self, outputs, targets):
        predictions = []
        references = []
        outputs = outputs.argmax(dim=2)
        
        for i in range(outputs.shape[0]):
            pred_tokens = []
            ref_tokens = []
            
            for j in range(outputs.shape[1]):
                idx = outputs[i, j].item()
                if idx == self.corpus.stoi["<EOS>"]: break
                if idx not in [0, 1]: pred_tokens.append(self.corpus.itos[idx])
            
            for j in range(targets.shape[1]):
                idx = targets[i, j].item()
                if idx not in [0, 1, 2]: ref_tokens.append(self.corpus.itos[idx])
                
            predictions.append(" ".join(pred_tokens))
            references.append([" ".join(ref_tokens)])
            
        return self.bleu_score(predictions, references)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_bleu"}}
