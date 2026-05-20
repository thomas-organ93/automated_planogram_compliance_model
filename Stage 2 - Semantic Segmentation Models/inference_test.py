import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from unet import UNet

# 1. Configuration
TEST_IMAGE_PATH = "test_image.png" # <-- Put your SKU-110K image path here
MODEL_WEIGHTS = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Model
model = UNet(in_channels=3, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.eval()

# 3. Load and Preprocess Image
img_bgr = cv2.imread(TEST_IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resize to prevent VRAM overflow (keeping it close to your 512x512 training scale)
img_resized = cv2.resize(img_rgb, (1024, 512))

# Match your exact training conversion logic
input_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(DEVICE)

# 4. Run Inference
print("Running prediction...")
with torch.no_grad():
    output = model(input_tensor)
    pred_map = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

# 5. Visualize Results side-by-side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_resized)
ax[0].set_title("Original Real-World Image")
ax[0].axis('off')

ax[1].imshow(pred_map * 255, cmap='gray')
ax[1].set_title("U-Net Prediction Mask")
ax[1].axis('off')

plt.tight_layout()
plt.show()