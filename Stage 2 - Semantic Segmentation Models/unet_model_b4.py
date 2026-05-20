import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import os
import glob
import matplotlib.pyplot as plt
import csv
from torch_lr_finder import LRFinder

# --- IMPORT UNET ---
from unet import UNet

# --- CONFIGURATION ---
IMAGE_PATH = sorted(glob.glob("./images/**/*.png", recursive=True))
MASK_PATH = sorted(glob.glob("./masks/**/*.png", recursive=True))

VAL_IMAGE_PATH = sorted(glob.glob("./val_images/**/*.png", recursive=True))
VAL_MASK_PATH = sorted(glob.glob("./val_masks/**/*.png", recursive=True))

# --- MODEL SAVE PATH ---
MODEL_SAVE_PATH = "unet_model.pth"

# Select Device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA (NVIDIA)")
else:
    DEVICE = "cpu"
    print("Using CPU (Slow)")


# ==========================================
# DATASET (TENSOR SLICING)
# ==========================================
class Dataset(Dataset):
    def __init__(self, image_path, mask_path, val):
        dataset_type = 'validation' if val else 'training'
        self.image_path = image_path
        self.mask_path = mask_path

        print(f"Total {dataset_type} images found: {len(self.image_path)}")
        print(f"Total {dataset_type} masks found: {len(self.mask_path)}\n")

        if len(self.image_path) != len(self.mask_path):
            raise ValueError("Image and mask paths must have the same length")

    def __len__(self):
        # double the size in regard to image 50% split
        return len(self.image_path) * 2

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_path)

        img_bgr = cv2.imread(self.image_path[real_idx])
        # force grayscale loading
        mask_gray = cv2.imread(self.mask_path[real_idx], cv2.IMREAD_GRAYSCALE)

        img_tensor = TF.to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Convert to tensor and immediately threshold to 0 and 1
        mask_tensor = torch.from_numpy(mask_gray).long()
        # Binary Threshold Tensor - True or False
        # It turns anything > 128 into 1, and everything else into 0.
        mask_tensor = (mask_tensor > 128).long()

        # 3. Deterministic 50% (1024x512 to 512x512) Crop
        h_total, w_total = img_tensor.shape[1], img_tensor.shape[2]
        crop_size = 512

        top = 0
        # top = np.random.randint(0, h_total - crop_size + 1)
        # left = np.random.randint(0, w_total - crop_size + 1)
        pass_number = idx // len(self.image_path)
        left = 0 if pass_number == 0 else (w_total - crop_size)

        img_crop = img_tensor[:, top:top + crop_size, left:left + crop_size]
        mask_crop = mask_tensor[top:top + crop_size, left:left + crop_size]

        # Final check: Ensure no values outside [0, 1]
        # (debugging)
        # if mask_crop.max() > 1:
        #     mask_crop = (mask_crop > 0).long()

        return img_crop, mask_crop


# ==========================================
# TRAINING LOOP
# ==========================================
def train(epochs, batch_size, lr, weights_list):
    train_acc_history = []
    val_acc_history = []
    loss_history = []
    val_miou_history = []
    background_iou_history = []
    foreground_iou_history = []

    best_val_loss = float("inf")
    # Setup Data
    train_dataset = Dataset(IMAGE_PATH, MASK_PATH, val=False)
    val_dataset = Dataset(VAL_IMAGE_PATH, VAL_MASK_PATH, val=True)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Randomizes image order
        num_workers=4, # Uses 4 CPU cores to "pre-fetch" images so the GPU never waits
        pin_memory=True, # Locks data into a "fast-lane" in RAM for quicker GPU transfer
        persistent_workers=True # Keep workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Keeps memory usage consistent with training
        num_workers=4, # Speeds up the validation check so training resumes faster
        shuffle=False, # No need to shuffle
        persistent_workers=True # Keep workers
    )


    # Setup Model
    model = UNet(in_channels=3, num_classes=2).to(DEVICE)

    # Load existing weights
    if os.path.isfile(MODEL_SAVE_PATH):
        print(f"Loading model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print(f"Creating model from scratch")

    # Weighting background at 2.0 or 3.0 forces the model to prioritize the gaps
    weights = torch.tensor(weights_list).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # Loss will go 'min' (down)
        factor=0.5, # Cut LR in half when stuck
        patience=4 # Wait 4 epoch of no improvement before cutting
    )

    print("\nStarting Training...")
    csv_filename = 'training_log.csv'
    writer_header = not os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if writer_header:
            writer.writerow(['Epoch', 'LR', 'Loss', 'TrainAcc', 'ValAcc', 'mIoU', 'Background_IoU', 'Foreground_IoU'])


    # Train on Epochs
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_loss, correct_train, total_pixels_train = 0, 0, 0
        accumulation_steps = 4  # 4 steps * Batch Size 4 = Virtual Batch Size 16. Gradient Accumulation
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Normalize the loss. Adding 4 losses together, so must divide by 4
            loss = loss / accumulation_steps

            # Backward pass (accumulate the gradients)
            loss.backward()
            # Step the optimizer only every 4 batches, or the end of dataset)
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()  # Clear memory for next virtual batch
            # Add the current batch loss to the total
            total_loss += loss.item() * accumulation_steps

            # Pixel-wise accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == masks).sum().item()
            total_pixels_train += masks.nelement()

        # Record Average Loss (Outside batch loop, inside epoch loop)
        avg_train_loss = total_loss / len(dataloader)
        loss_history.append(avg_train_loss)
        train_acc = (correct_train / total_pixels_train) * 100

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct_val, total_pixels_val = 0, 0, 0
        val_iou = np.array([0.0, 0.0])

        with torch.no_grad():  # Essential: Saves memory and stops learning
            for v_images, v_masks in val_dataloader:  # Use your separate val_loader
                v_images, v_masks = v_images.to(DEVICE), v_masks.to(DEVICE)

                v_outputs = model(v_images)
                v_loss = criterion(v_outputs, v_masks)
                val_loss += v_loss.item()

                _, v_predicted = torch.max(v_outputs, 1)
                correct_val += (v_predicted == v_masks).sum().item()
                total_pixels_val += v_masks.nelement()

                # Calculate IoU for this batch and add it to the running total
                batch_iou = calculate_iou_metrics(v_predicted, v_masks)
                val_iou += np.array(batch_iou)

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = (correct_val / total_pixels_val) * 100

        # Calculate Final IoU averages for the epoch
        avg_iou = (val_iou / len(val_dataloader)) * 100
        miou = np.mean(avg_iou)  # This is mean intersection over union (mIoU)
        background_iou = avg_iou[0]  # Performance on background/gaps
        foreground_iou = avg_iou[1]  # Performance on the foreground products/instances

        # Step scheduler based on VALIDATION loss
        scheduler.step(avg_val_loss)

        # Store for plotting later
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        val_miou_history.append(miou)
        background_iou_history.append(background_iou)
        foreground_iou_history.append(foreground_iou)
        current_lr = optimizer.param_groups[0]['lr']

        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            display_validation_image(model, "best_prediction_results")

        # --SAVE LOG--
        # Prepare data row
        log_data = [epoch + 1, current_lr, avg_train_loss, train_acc, val_acc, miou, background_iou, foreground_iou]
        # Append to file (creates the file if it doesn't exist)
        with open('training_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        print(f"Epoch {epoch + 1}| Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | mIoU: {miou:.2f}% | Background mIoU: {background_iou:.2f}% | Foreground IoU: {foreground_iou:.2f}%")


    # Save in general latest
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Finished. Model saved to {MODEL_SAVE_PATH}")

    # Run Final Test
    display_validation_image(model, "prediction_results")

    return train_acc_history, val_acc_history, loss_history, val_miou_history, background_iou_history, foreground_iou_history


# ==========================================
# FULL RESOLUTION INFERENCE
# ==========================================
def display_validation_image(model, dir_name):
    print("\nRunning Inference on 5 validation images...")
    model.eval()
    os.makedirs(dir_name, exist_ok=True)
    # Only 5 example images
    for paths in VAL_IMAGE_PATH[:5]:
        img_bgr = cv2.imread(paths)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            pred_map = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        base_name = os.path.splitext(os.path.basename(paths))[0]
        cv2.imwrite(f"{dir_name}/{base_name}.png", pred_map * 255)
    print(f"Success! Saved '{dir_name}' folder.")

def calculate_iou_metrics(predicted, target, num_classes=2):
    # Calculate the Intersection over Union (IoU)
    iou_per_class = []
    epsilon = 1e-6
    for cls in range(num_classes):
        pred_idx = (predicted == cls)
        target_idx = (target == cls)
        intersection = (pred_idx & target_idx).sum().item()
        union = (pred_idx | target_idx).sum().item()
        iou_per_class.append((intersection + epsilon) / (union + epsilon))
    return iou_per_class


if __name__ == "__main__":
    WEIGHTS = [5.0, 1.0]
    EPOCH = 30
    BATCH_SIZE = 4
    LR = 2.48E-04
    train_acc_hist, val_acc_hist, loss_hist, val_miou_hist, background_iou_hist, foreground_iou_hist = train(EPOCH, BATCH_SIZE, LR, WEIGHTS)

    # --- 4. VISUALIZATION ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 18))
    ax[0].plot(train_acc_hist, label='Training Accuracy')
    ax[0].plot(val_acc_hist, label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].legend()

    ax[1].plot(loss_hist, label='Training Loss', color='blue')
    ax[1].set_title('Loss Record')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    ax[2].plot(val_miou_hist, label='Mean IoU (mIoU)', color='tab:green', linestyle='--')
    ax[2].plot(background_iou_hist, label='Background/Gaps IoU (Class 0)', color='tab:purple', alpha=0.6)
    ax[2].plot(foreground_iou_hist, label='Foreground/Instances IoU (Class 1)', color='tab:red', alpha=1.0)
    ax[2].set_title('Segmentation Quality (mIoU, Background & Foreground Preservation)')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('IoU (%)')
    ax[2].legend()

    plt.subplots_adjust(hspace=0.4)
    plt.suptitle('U-Net Testing', fontsize=16)
    plt.show()

    '''
    #Debugging#:
    Learning Rate: 2.48E-04 (Recommended by PyTorch LR Finder).
    Resolution: 512 x 512.
    Weights: [5.0, 1.0] (Background weight is 5.0).
    Batch Size: 4 (Fastest to initialise).
    
    --Calculate Settings--
    Overall images = 1,920
    Virtual Length (return): 1,920 x 2 = 3,840
    Steps: 3,840 / 4 (batch) = 960 steps
    Total Steps needed: 20,000 (required for custom model)
    Epoch Calc: 20,000 / 960 ≈ 20 Epochs (Round up to 25) = 25 Epoch
    Scheduler Patience = 2 (drop LR after 2 bad epochs)
    
    -----------------------------------------------------------
    
    #Training#:
    Learning Rate: 2.48E-04 (Recommended by PyTorch LR Finder).
    Resolution: 512 x 512.
    Weights: [5.0, 1.0] (Background weight is 5.0).
    Batch Size: 4 (Fastest to initialise).
    
    --Calculate Settings--
    Overall images = 27,600
    Virtual Length (return): 27,600 x 2 = 55,200
    Steps: 55,200 / 4 (batch) = 13,800 steps
    Total Steps needed: 400,000 (required for custom model)
    Epoch Calc: 400,000 / 13,800 ≈ 28 Epochs (Round up to 30) = 30 Epoch
    Scheduler Patience = 4 (drop LR after 4 bad epochs)
    '''

# if __name__ == "__main__":
#     WEIGHTS = [5.0, 1.0]
#     BATCH_SIZE = 4
#
#     temp_model = UNet(in_channels=3, num_classes=2).to(DEVICE)
#     temp_optimiser = optim.Adam(temp_model.parameters(), lr=1e-7)
#     temp_weights = torch.tensor(WEIGHTS).to(DEVICE)
#     temp_criterion = nn.CrossEntropyLoss(weight=temp_weights)
#
#     # Run LR Finder
#     lr_dataset = Dataset(IMAGE_PATH, MASK_PATH, val=False)
#     lr_loader = DataLoader(lr_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     print("Running Learning Rate Finder...")
#     lr_finder = LRFinder(temp_model, temp_optimiser, temp_criterion, device=DEVICE)
#
#     # num_iter=100 will automatically cycle through 150 virtual crops
#     # about 10-11 times to get enough data for a smooth plot.
#     lr_finder.range_test(lr_loader, end_lr=10, num_iter=100)
#     lr_finder.plot()
#     lr_finder.reset()
#
#     # Update after looking at plot
#     LR = 0.0001
#     EPOCH = 60
#     # train()

'''
    PyTorch Learning Rate Finder (LR Finder)
    - Is a diagnostic tool used to determine the optimal learning rate (LR) for a model without having to perform
    dozens of trail-and-error training runs.
    Mechanism
    - Executes and miniature 'mock' training session (usally about 70-100 iterations)
    - Starts off with extremely small learning rate.
    - Increases LR exponentially until it reaches a very high value (e.g., 1 or 10).
    - Eventually, the LR becomes so high that the model's weights explode. Here the tool stops the test.
    - Result: 
'''


