from torch_lr_finder import LRFinder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import deep_watershed_transform_network as dwt_network
import direction_network_model as dn_model
import deep_watershed_dataset as wtn_dataset
import glob

# --- 1. Device Setup ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Success: Using Apple Metal (MPS) acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA (NVIDIA).")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (Warning: This will be slower).")

IMAGE_PATH = sorted(glob.glob("./images/**/*.jpg", recursive=True))
MASK_PATH = sorted(glob.glob("./masks/**/*.jpg", recursive=True))
NPZ_TRAINING_PATH = "./training_data_npz"

# ==========================================
# DEFINE WRAPPERS
# ==========================================

class DN_LR_Wrapper(Dataset):
    """Feeds (Image, Direction) to the Direction Network"""
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        # Unpack the 3 items
        inputs, direction, energy = self.ds[idx]
        # Return only what DN needs
        return inputs, direction

class WTN_LR_Wrapper(Dataset):
    """Feeds (Direction, Energy) to the Watershed Network"""
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        # Unpack the 3 items
        inputs, direction, energy = self.ds[idx]
        # Return only what WTN needs
        return direction, energy

# ==========================================
# RUN BOTH FINDERS
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 4

    # --- Load Base Dataset Once ---
    base_dataset = wtn_dataset.DeepWatershedDataset(IMAGE_PATH, MASK_PATH, NPZ_TRAINING_PATH, val=False)

    # -------------------------------------------------------
    # PHASE 1: Find LR for Direction Network (DN)
    # -------------------------------------------------------
    print("\n>>> Running LRFinder for DIRECTION NETWORK (DN)...")

    # A. Setup Model
    dn_model = dn_model.DirectionNetwork().to(DEVICE)
    dn_optim = optim.Adam(dn_model.parameters(), lr=1e-7)
    dn_crit = nn.MSELoss()

    # B. Setup Data (Use DN Wrapper!)
    dn_loader = DataLoader(DN_LR_Wrapper(base_dataset), batch_size=BATCH_SIZE, shuffle=True)

    # C. Run Test
    dn_finder = LRFinder(dn_model, dn_optim, dn_crit, device=DEVICE)
    dn_finder.range_test(dn_loader, end_lr=10, num_iter=100)

    # D. Suggestion
    lrs = dn_finder.history["lr"]
    losses = dn_finder.history["loss"]
    grad = np.gradient(losses)
    suggested_lr_dn = lrs[np.argmin(grad)]
    print(f"--- DN Suggestion: {suggested_lr_dn:.6f} ---")

    # E. Plot
    dn_finder.plot()
    plt.show()
    dn_finder.reset()

    # -------------------------------------------------------
    # PHASE 2: Find LR for Watershed Network (WTN)
    # -------------------------------------------------------
    print("\nRunning LRFinder for WATERSHED NETWORK (WTN)...")

    # A. Setup Model
    wtn_model = dwt_network.WatershedTransformNetwork().to(DEVICE)
    wtn_optim = optim.Adam(wtn_model.parameters(), lr=1e-7)

    # Weights for Class Imbalance
    wtn_weights = torch.ones(16).to(DEVICE)
    wtn_weights[0] = 0.05
    wtn_crit = nn.CrossEntropyLoss(weight=wtn_weights)

    # B. Setup Data (Use WTN Wrapper!)
    wtn_loader = DataLoader(WTN_LR_Wrapper(base_dataset), batch_size=BATCH_SIZE, shuffle=True)

    # C. Run Test
    wtn_finder = LRFinder(wtn_model, wtn_optim, wtn_crit, device=DEVICE)
    wtn_finder.range_test(wtn_loader, end_lr=10, num_iter=100)

    # D. Suggestion
    lrs = wtn_finder.history["lr"]
    losses = wtn_finder.history["loss"]
    grad = np.gradient(losses)
    suggested_lr_wtn = lrs[np.argmin(grad)]
    print(f"--- WTN Suggestion: {suggested_lr_wtn:.6f} ---")

    # E. Plot
    wtn_finder.plot()
    plt.show()
    wtn_finder.reset()

    '''
    --- DN Suggestion: 0.000248 ---
    LR suggestion: steepest gradient
    Suggested LR: 2.48E-04
    3e-4
    
    --- WTN Suggestion: 0.291505 ---
    LR suggestion: steepest gradient
    Suggested LR: 2.92E-01 (bad)
    
    1e-3 (The standard "Golden Rule" for Adam Classification).
    '''