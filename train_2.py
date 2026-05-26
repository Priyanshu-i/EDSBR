# %% [markdown]
# # Phase 3: Production Training - Embodied Latent Adapter (Delta Model)
# This notebook trains the final Autoregressive Continuous Transformer on a 50,000-sample dataset.
# The network translates semantic text embeddings (CLIP) into normalized continuous Delta vectors: 
# `[x1, y1, dx_norm, dy_norm, thickness, opacity]`.
# 
# **Optimized for 4GB VRAM (RTX 2050):** Utilizes PyTorch 2.0 AMP, Gradient Accumulation, and JSONL streaming.

# %%
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")
if torch.cuda.is_available():
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# %% [markdown]
# ## 1. JSONL Dataset Loader & Sequence Padding
# Because 50,000 samples can consume significant memory, we load the JSONL file cleanly. 
# We pad all sequences to a maximum length (e.g., 80) to accommodate the most complex architecture diagrams in the dataset.

# %%
class StrokeJSONLDataset(Dataset):
    def __init__(self, jsonl_path, max_seq_len=80):
        self.max_seq_len = max_seq_len
        self.data = []
        
        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Successfully loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        
        # Strokes are already sorted and delta-normalized by the factory script
        # Format: [x1, y1, dx_norm, dy_norm, w, a]
        strokes = torch.tensor(item["strokes"], dtype=torch.float32)
        
        # Truncate if unexpectedly long
        if strokes.shape[0] > self.max_seq_len:
            strokes = strokes[:self.max_seq_len]
            
        # Pad sequence with zeros
        pad_len = self.max_seq_len - strokes.shape[0]
        if pad_len > 0:
            padding = torch.zeros((pad_len, 6), dtype=torch.float32)
            strokes = torch.cat([strokes, padding], dim=0)
            
        return prompt, strokes

# Initialize DataLoader
MAX_STROKES = 80
BATCH_SIZE = 4 # Kept small for 4GB VRAM
dataset = StrokeJSONLDataset("dataset.jsonl", max_seq_len=MAX_STROKES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %% [markdown]
# ## 2. The Delta Continuous Transformer
# The model maps 512D text embeddings into the continuous geometric space. 
# It predicts relative deltas ($\Delta x$, $\Delta y$) which allows the AI to draw fluid, connected shapes.

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].transpose(0, 1)

class DeltaContinuousModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. Text Encoder (Frozen)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # 2. Continuous Embedder
        self.stroke_embedding = nn.Linear(6, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Autoregressive Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, 
            batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. Output Head (Bounds all outputs strictly between 0.0 and 1.0)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 6),
            nn.Sigmoid() 
        )

    def generate_causal_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, text_prompts, target_strokes):
        batch_size, seq_len, _ = target_strokes.shape
        
        text_inputs = self.tokenizer(text_prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(**text_inputs).last_hidden_state
            
        stroke_embeds = self.stroke_embedding(target_strokes)
        sos_expanded = self.sos_token.expand(batch_size, -1, -1)
        decoder_inputs = torch.cat([sos_expanded, stroke_embeds[:, :-1, :]], dim=1)
        decoder_inputs = self.pos_encoder(decoder_inputs)
        
        tgt_mask = self.generate_causal_mask(seq_len, device)
        hidden_states = self.transformer_decoder(
            tgt=decoder_inputs, memory=encoder_hidden_states, tgt_mask=tgt_mask
        )
        
        return self.output_head(hidden_states)

model = DeltaContinuousModel().to(device)
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 3. Production Training Loop
# Utilizing PyTorch 2.0+ specific AMP (`torch.amp.autocast`) and Gradient Accumulation to 
# ensure smooth scaling across 50,000 samples without memory crashes.

# %%
def train_production_model(model, dataloader, epochs=50, save_path="delta_model_v1.pth"):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Updated PyTorch 2.0+ AMP syntax
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = 16 # Simulates a true batch size of 64 (16 * 4)
    
    model.train()
    print("Beginning 50k Full-Scale Training...")
    
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for batch_idx, (prompts, targets) in enumerate(dataloader):
            targets = targets.to(device)
            
            with torch.amp.autocast('cuda'):
                predictions = model(prompts, targets)
                loss = criterion(predictions, targets) / accumulation_steps
                
            scaler.scale(loss).backward()
            total_loss += (loss.item() * accumulation_steps)
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | MSE Loss: {avg_loss:.6f}")
        
        # Save checkpoints periodically to prevent data loss
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    # Save final production weights
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete. Weights saved to {save_path}")
    
    plt.figure(figsize=(6, 3))
    plt.plot(loss_history, color='green', lw=2)
    plt.title("Production Training Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

# %% [markdown]
# ## 4. Inference & Delta Denormalization Renderer
# Generates the vectors autoregressively and reverses the mathematics:
# it translates the predicted `dx_norm` and `dy_norm` back into absolute 
# ending coordinates (`x2`, `y2`) so they can be plotted accurately on the canvas.

# %%
def generate_validation_gallery(model, prompts, max_strokes=80, cols=3):
    model.eval()
    n_samples = len(prompts)
    rows = math.ceil(n_samples / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    print("\nGenerating Production Validation Gallery...")
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            text_inputs = model.tokenizer([prompt], padding=True, return_tensors="pt").to(device)
            encoder_hidden_states = model.text_encoder(**text_inputs).last_hidden_state
            
            current_strokes = torch.zeros((1, 0, 6), device=device)
            
            for step in range(max_strokes):
                if current_strokes.size(1) > 0:
                    stroke_embeds = model.stroke_embedding(current_strokes)
                    decoder_inputs = torch.cat([model.sos_token, stroke_embeds], dim=1)
                else:
                    decoder_inputs = model.sos_token
                    
                decoder_inputs = model.pos_encoder(decoder_inputs)
                tgt_mask = model.generate_causal_mask(decoder_inputs.size(1), device)
                
                hidden_states = model.transformer_decoder(
                    tgt=decoder_inputs, memory=encoder_hidden_states, tgt_mask=tgt_mask
                )
                
                next_stroke_pred = model.output_head(hidden_states[:, -1:, :])
                current_strokes = torch.cat([current_strokes, next_stroke_pred], dim=1)
                
                # Dynamic Early Stopping: If opacity drops to 0, stop drawing.
                if next_stroke_pred[0, 0, 5].item() < 0.1:
                    break
                
            generated_vectors = current_strokes.squeeze(0).cpu().numpy()
            
            # --- Rendering Phase ---
            ax = axes[idx]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_title(prompt[:40], fontsize=10)
            
            for stroke in generated_vectors:
                x1, y1, dx_norm, dy_norm, thickness, opacity = stroke
                
                if opacity > 0.3:
                    # Reverse the normalization math to get absolute coordinates
                    x2 = x1 + (dx_norm * 2.0 - 1.0)
                    y2 = y1 + (dy_norm * 2.0 - 1.0)
                    
                    lw = max(1.0, thickness * 20.0)
                    ax.plot([x1, x2], [y1, y2], color='black', alpha=opacity, linewidth=lw, solid_capstyle='round')
            
            ax.axis('off')

    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Execution Block

# %%
if __name__ == "__main__":
    # 1. Run the Training (Set epochs=30 or 50 based on convergence speed)
    train_production_model(model, dataloader, epochs=50)
    
    # 2. Test unseen real-world prompts
    test_prompts = [
        "Draw a load balancer routing to 3 server nodes",
        "Draw a flowchart with 2 decision trees",
        "Draw a neural network star topology",
        "Draw a browser UI login wireframe",
        "Draw a concentric geometric cube",
        "Draw a data pipeline entering a cylinder database"
    ]
    
    generate_validation_gallery(model, test_prompts, max_strokes=MAX_STROKES, cols=3)