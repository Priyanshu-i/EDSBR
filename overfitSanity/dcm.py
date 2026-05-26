import os
import math
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

local_path = "./clip-vit-base-patch32-local"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].transpose(0, 1)


class DeltaContinuousModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        # 1. Frozen Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(local_path)
        self.text_encoder = CLIPTextModel.from_pretrained(local_path)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 2. Continuous Embedder & Context Projection
        self.stroke_embedding = nn.Linear(6, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # FIX: Project the CLIP pooled output directly into the sequence 
        # to act as a semantically-rich Start-of-Sequence token.
        text_hidden_size = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(text_hidden_size, d_model)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 4. Regression Output Head
        # FIX: Removed Sigmoid. We now predict unbounded Z-score normalized logits.
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 6)
        )

    def generate_causal_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

    def forward(self, text_prompts, target_strokes):
        batch_size, seq_len, _ = target_strokes.shape
        device = next(self.parameters()).device

        # Encode Text
        text_inputs = self.tokenizer(
            text_prompts, padding=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            encoder_outputs = self.text_encoder(**text_inputs)
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
        # FIX: Use the [CLS] token as the starting semantic context
        cls_token = encoder_hidden_states[:, 0, :]  # Shape: [B, hidden_size]
        context_embed = self.text_projection(cls_token).unsqueeze(1) # Shape: [B, 1, d_model]

        # Embed previous strokes
        stroke_embeds = self.stroke_embedding(target_strokes)
        
        # FIX: Shift right for teacher forcing, prepending the text context
        decoder_inputs = torch.cat([context_embed, stroke_embeds[:, :-1, :]], dim=1)
        decoder_inputs = self.pos_encoder(decoder_inputs)

        # Autoregressive decoding
        tgt_mask = self.generate_causal_mask(seq_len, device)
        hidden_states = self.transformer_decoder(
            tgt=decoder_inputs, memory=encoder_hidden_states, tgt_mask=tgt_mask
        )

        return self.output_head(hidden_states)