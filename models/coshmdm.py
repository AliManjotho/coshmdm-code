import torch
import clip

from torch import nn
from models import *


class CoShMDM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        bert_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = bert_model.token_embedding
        self.bert_transformer = bert_model.transformer
        self.positional_embedding = bert_model.positional_embedding
        self.ln_final = bert_model.ln_final
        self.dtype = bert_model.dtype

        set_requires_grad(self.bert_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        bertTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.bertTransEncoder = nn.TransformerEncoder(
            bertTransEncoderLayer,
            num_layers=2)
        self.bert_ln = nn.LayerNorm(768)

    def compute_loss(self, batch):
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        device = next(self.bert_transformer.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.bert_transformer(x)
            x = x.permute(1, 0, 2)
            bert_out = self.ln_final(x).type(self.dtype)

        out = self.bertTransEncoder(bert_out)
        out = self.bert_ln(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch["cond"] = cond

        return batch
