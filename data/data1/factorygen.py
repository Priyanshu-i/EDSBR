import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import scipy.ndimage as nd
import os
import json
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. The Differentiable Canvas ---
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
        gx, gy = self.grid_x.unsqueeze(0), self.grid_y.unsqueeze(0)
        
        for i in range(stroke_params.shape[0]):
            x1, y1, x2, y2, thickness, opacity = stroke_params[i]
            ax, ay = x2 - x1, y2 - y1
            mag_sq = ax**2 + ay**2 + 1e-6
            px, py = gx - x1, gy - y1
            t = torch.clamp((px * ax + py * ay) / mag_sq, 0.0, 1.0)
            closest_x, closest_y = x1 + t * ax, y1 + t * ay
            dist_sq = (gx - closest_x)**2 + (gy - closest_y)**2
            sigma = thickness * 0.02 + global_blur
            stroke = opacity * torch.exp(-dist_sq / (2 * sigma**2))
            canvas = canvas - stroke
            
        return torch.clamp(canvas, min=0.0, max=1.0)

# --- 2. Dynamic Heuristics Analyzer ---
def analyze_image_heuristics(img_array):
    """
    Analyzes a binary target image to intelligently guess the required NUM_STROKES
    and dynamic parameters.
    """
    # Convert to 8-bit for OpenCV
    img_8u = (img_array * 255).astype(np.uint8)
    
    # Invert so background is 0, lines are 255
    inverted = cv2.bitwise_not(img_8u)
    
    # Skeletonize/Canny to find structural edges
    edges = cv2.Canny(inverted, 50, 150)
    total_edge_pixels = np.sum(edges > 0)
    
    # Heuristic: Assume an average stroke covers ~20 pixels in a 128x128 canvas
    # Add a 30% surplus because V3's opacity penalty will delete the extras
    estimated_strokes = int((total_edge_pixels / 20) * 1.3)
    
    # Bounding values to prevent OOM or undertraining
    num_strokes = max(5, min(estimated_strokes, 100))
    
    # If the image is highly complex, start with a slightly smaller blur so 
    # it doesn't just turn into a single grey blob.
    start_blur = 0.15 if num_strokes < 30 else 0.08
    
    return num_strokes, start_blur

# --- 3. The Headless Optimizer ---
def optimize_image(image_path, canvas_size=128, epochs=1200):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    img = cv2.resize(img, (canvas_size, canvas_size))
    # Binarize to crisp 0.0 or 1.0
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    target_img = (img / 255.0).astype(np.float32)
    
    # Calculate Dynamic Variables
    num_strokes, start_blur = analyze_image_heuristics(target_img)
    end_blur = 0.005
    
    print(f"[{os.path.basename(image_path)}] Dynamically assigned {num_strokes} strokes.")
    
    target_tensor = torch.tensor(target_img, device=device)
    
    # SDF Computation
    background_mask = (target_img > 0.5)
    distance_field = nd.distance_transform_edt(background_mask) / canvas_size
    sdf_tensor = torch.tensor(distance_field, device=device, dtype=torch.float32)
    
    # Setup V3 Engine
    renderer = DifferentiableCanvasV3(canvas_size=canvas_size)
    raw_stroke_params = torch.randn((num_strokes, 6), device=device, requires_grad=True)
    optimizer = optim.Adam([raw_stroke_params], lr=0.05)
    
    LAMBDA_SDF, LAMBDA_LEN, LAMBDA_OPAC = 0.5, 0.05, 0.005

    # Optimization Loop
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        progress = epoch / epochs
        current_blur = start_blur * ((end_blur / start_blur) ** progress)
        scaled_params = torch.sigmoid(raw_stroke_params)
        rendered_canvas = renderer.draw_lines(scaled_params, global_blur=current_blur)
        
        mse_loss = nn.MSELoss()(rendered_canvas, target_tensor)
        
        current_sdf_weight = LAMBDA_SDF * min(1.0, (epoch - epochs // 3) / (epochs // 3)) if epoch > epochs // 3 else 0.0
        sdf_loss = torch.mean((1.0 - rendered_canvas) * sdf_tensor)
        
        dx = scaled_params[:, 2] - scaled_params[:, 0]
        dy = scaled_params[:, 3] - scaled_params[:, 1]
        lengths = torch.sqrt(dx**2 + dy**2 + 1e-6)
        length_loss = torch.mean(lengths * scaled_params[:, 5])
        
        polarization = 0.1 * torch.mean(scaled_params[:, 5] * (1.0 - scaled_params[:, 5]))
        opacity_pen = LAMBDA_OPAC * torch.mean(scaled_params[:, 5])
        
        total_loss = (10.0 * mse_loss) + (current_sdf_weight * sdf_loss) + (LAMBDA_LEN * length_loss) + opacity_pen + polarization
        total_loss.backward()
        optimizer.step()

    # Final Extraction and Cleanup
    with torch.no_grad():
        final_params = torch.sigmoid(raw_stroke_params).cpu().numpy()
        
    # Filter out "erased" strokes (Opacity < 0.5)
    active_strokes = final_params[final_params[:, 5] > 0.5]
    
    return active_strokes.tolist()

# --- 4. Batch Execution ---
def run_factory(input_dir="data/images", output_file="stroke_dataset.json"):
    dataset = []
    image_files = glob.glob(os.path.join(input_dir, "*.png")) # Add .jpg if needed
    
    for img_path in image_files:
        strokes = optimize_image(img_path)
        if strokes:
            dataset.append({
                "file": os.path.basename(img_path),
                "strokes": strokes # [x1, y1, x2, y2, thickness, opacity]
            })
            
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Factory complete. Processed {len(dataset)} images.")

# To execute:
# run_factory(input_dir="./my_raster_sketches")