# %% [markdown]
# # V3 Embodied Canvas Agent: Geometric Edge Fitter
# Upgraded with Signed Distance Fields (SDF), Stroke Length Regularization, 
# and Delayed Opacity Penalties to enforce structural exactness.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as nd

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. The Differentiable Canvas
# (Unchanged from V2. Maps continuous vectors to pixel intensities).

# %%
class DifferentiableCanvasV3(nn.Module):
    def __init__(self, canvas_size=128):
        super().__init__()
        self.canvas_size = canvas_size
        y, x = torch.meshgrid(
            torch.linspace(0, 1, canvas_size, device=device),
            torch.linspace(0, 1, canvas_size, device=device),
            indexing='ij'
        )
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)

    def draw_lines(self, stroke_params, global_blur):
        canvas = torch.ones((self.canvas_size, self.canvas_size), device=device)
        gx = self.grid_x
        gy = self.grid_y
        
        for i in range(stroke_params.shape[0]):
            x1, y1, x2, y2, thickness, opacity = stroke_params[i]
            
            ax, ay = x2 - x1, y2 - y1
            mag_sq = ax**2 + ay**2 + 1e-6
            
            px, py = gx - x1, gy - y1
            t = torch.clamp((px * ax + py * ay) / mag_sq, 0.0, 1.0)
            
            closest_x = x1 + t * ax
            closest_y = y1 + t * ay
            dist_sq = (gx - closest_x)**2 + (gy - closest_y)**2
            
            sigma = thickness * 0.02 + global_blur
            stroke = opacity * torch.exp(-dist_sq / (2 * sigma**2))
            
            canvas = canvas - stroke
            
        return torch.clamp(canvas, min=0.0, max=1.0)

# %% [markdown]
# ## 2. Target Generation & The Signed Distance Field (SDF)
# We generate the target image, but now we also compute a "Distance Map". 
# Every white pixel gets a mathematical value representing how far it is from a black line.
# If the AI draws on a high-value pixel, it receives a massive penalty.

# %%
CANVAS_SIZE = 128
target_img = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

# Star Target
cv2.line(target_img, (50, 10), (60, 40), 0.0, 3)    # stroke 1
cv2.line(target_img, (60, 40), (90, 40), 0.0, 3)    # stroke 2
cv2.line(target_img, (90, 40), (65, 60), 0.0, 3)    # stroke 3
cv2.line(target_img, (65, 60), (75, 90), 0.0, 3)    # stroke 4
cv2.line(target_img, (75, 90), (50, 70), 0.0, 3)    # stroke 5
cv2.line(target_img, (50, 70), (25, 90), 0.0, 3)    # stroke 6
cv2.line(target_img, (25, 90), (35, 60), 0.0, 3)    # stroke 7
cv2.line(target_img, (35, 60), (10, 40), 0.0, 3)    # stroke 8
cv2.line(target_img, (10, 40), (40, 40), 0.0, 3)    # stroke 9
cv2.line(target_img, (40, 40), (50, 10), 0.0, 3)    # stroke 10

# NUM_STROKES = 10


target_tensor = torch.tensor(target_img, device=device)

# Create the Empty Space Penalty Map (Signed Distance Field)
# boolean mask where background (white) is True
background_mask = (target_img > 0.5) 
# Calculate distance from each True pixel to the nearest False (black) pixel
distance_field = nd.distance_transform_edt(background_mask)

# Normalize the distances so gradients don't explode
distance_field = distance_field / CANVAS_SIZE
sdf_tensor = torch.tensor(distance_field, device=device, dtype=torch.float32)

# Visualize the SDF Field
plt.figure(figsize=(4, 4))
plt.imshow(sdf_tensor.cpu().numpy(), cmap='magma')
plt.title("Empty Space Penalty Map (SDF)\nBright = High Penalty")
plt.axis('off')
plt.show()

# %% [markdown]
# ## 3. The Geometric Optimization Loop
# We combine Pixel MSE, SDF Penalty, Length Regularization, and a delayed Opacity Penalty.

# %%
renderer = DifferentiableCanvasV3(canvas_size=CANVAS_SIZE)
NUM_STROKES = 10
raw_stroke_params = torch.randn((NUM_STROKES, 6), device=device, requires_grad=True)

optimizer = optim.Adam([raw_stroke_params], lr=0.05)

epochs = 1200
start_blur = 0.15
end_blur = 0.005

# Loss weights to balance the objectives
LAMBDA_SDF = 0.5    # Penalty for drawing in empty space
LAMBDA_LEN = 0.05   # Penalty for very long strokes
LAMBDA_OPAC = 0.005  # Penalty for using unnecessary strokes

print("Starting Geometric Edge Optimization...")
for epoch in range(epochs + 1):
    optimizer.zero_grad()
    
    progress = epoch / epochs
    current_blur = start_blur * ((end_blur / start_blur) ** progress)
    scaled_params = torch.sigmoid(raw_stroke_params)
    rendered_canvas = renderer.draw_lines(scaled_params, global_blur=current_blur)
    
    # 1. Pixel-Wise Loss
    mse_loss = nn.MSELoss()(rendered_canvas, target_tensor)
    
    # 2. SDF Penalty (Annealed)
    # Give the strokes 400 epochs to travel across the canvas freely, 
    # then slowly turn on the forcefield to snap them cleanly into the lines.
    current_sdf_weight = 0.0
    if epoch > epochs // 3:
        ramp = min(1.0, (epoch - epochs // 3) / (epochs // 3))
        current_sdf_weight = LAMBDA_SDF * ramp
        
    ink_rendered = 1.0 - rendered_canvas
    sdf_loss = torch.mean(ink_rendered * sdf_tensor)
    
    # 3. Stroke Length Penalty
    dx = scaled_params[:, 2] - scaled_params[:, 0]
    dy = scaled_params[:, 3] - scaled_params[:, 1]
    lengths = torch.sqrt(dx**2 + dy**2 + 1e-6)
    length_loss = torch.mean(lengths * scaled_params[:, 5])
    
    # 4. Opacity Polarization & Penalty
    # Forces opacity to snap to either 1.0 (ink) or 0.0 (erased), destroying the "stacking" loophole.
    polarization_penalty = 0.1 * torch.mean(scaled_params[:, 5] * (1.0 - scaled_params[:, 5]))
    opacity_penalty = LAMBDA_OPAC * torch.mean(scaled_params[:, 5])
    
    # Total Composite Loss
    total_loss = (10.0 * mse_loss) + (current_sdf_weight * sdf_loss) + \
                 (LAMBDA_LEN * length_loss) + opacity_penalty + polarization_penalty

    total_loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.5f} | "
              f"MSE: {mse_loss.item():.4f} | SDF: {sdf_loss.item():.4f}")

# %% [markdown]
# ## 4. Final Verification

# %%
with torch.no_grad():
    final_params = torch.sigmoid(raw_stroke_params)
    final_canvas = renderer.draw_lines(final_params, global_blur=end_blur).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(target_tensor.cpu().numpy(), cmap='gray')
axes[0].set_title("Target Infrastructure Map")
axes[0].axis('off')

axes[1].imshow(final_canvas, cmap='gray')
axes[1].set_title(f"AI Stroke Recovery (Geometrically Enforced)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Print Opacities to verify unused strokes were deleted
final_opacities = final_params[:, 5].detach().cpu().numpy()
print("\nFinal Stroke Opacities (Notice which ones dropped near 0.0):")
print(np.round(final_opacities, 3))