import numpy as np
import random
import os
import cv2

def get_random_distractor(coco_obj, exclude_cat_id, img_dir):
    """
    Fetches a random crop and mask from a category
    """
    # Get all category IDs
    all_cat_ids = coco_obj.getCatIds()

    # Filter out the current product's category
    distractor_cats = [c for c in all_cat_ids if c != exclude_cat_id]

    # Retry loop in case we pick a bad/empty image
    for _ in range(5):
        if not distractor_cats: break # Safety check

        # Pick random category -> random image -> random annotation
        rand_cat = random.choice(distractor_cats)
        img_ids = coco_obj.getImgIds(catIds=[rand_cat])

        if not img_ids: continue

        rand_img_id = random.choice(img_ids)
        rand_ann_ids = coco_obj.getAnnIds(imgIds=rand_img_id)

        if not rand_ann_ids: continue

        # Load data
        rand_ann = coco_obj.loadAnns(random.choice(rand_ann_ids))[0]
        img_info = coco_obj.loadImgs(rand_img_id)[0]
        full_path = os.path.join(img_dir, img_info['file_name'])

        original_img = cv2.imread(full_path)
        if original_img is None: continue

        # --- CROP LOGIC ---
        mask = coco_obj.annToMask(rand_ann) * 255
        mask = mask.astype(np.uint8)

        x, y, w, h = [int(val) for val in rand_ann['bbox']]
        h_img, w_img = original_img.shape[:2]
        y1, y2 = max(0, y), min(h_img, y + h)
        x1, x2 = max(0, x), min(w_img, x + w)

        crop_img = original_img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        if crop_img.size > 0 and crop_mask.size > 0:
            return crop_img, crop_mask

    return None, None # Failed to find a distractor

def get_random_crop_from_category(coco_obj, target_cat_id, img_dir):
    """
    Fetches a random crop and mask from the SPECIFIED category.
    """
    # Get all images that contain this product
    img_ids = coco_obj.getImgIds(catIds=[target_cat_id])

    for _ in range(5):
        if not img_ids: break

        # Pick random image -> random annotation
        rand_img_id = random.choice(img_ids)

        ann_ids = coco_obj.getAnnIds(imgIds=rand_img_id, catIds=[target_cat_id])
        if not ann_ids: continue

        rand_ann = coco_obj.loadAnns(random.choice(ann_ids))[0]

        # Load Image
        img_info = coco_obj.loadImgs(rand_img_id)[0]
        full_path = os.path.join(img_dir, img_info['file_name'])
        original_img = cv2.imread(full_path)

        if original_img is None: continue

        # --- STANDARD CROP LOGIC ---
        mask = coco_obj.annToMask(rand_ann) * 255
        mask = mask.astype(np.uint8)

        x, y, w, h = [int(val) for val in rand_ann['bbox']]
        h_img, w_img = original_img.shape[:2]
        y1, y2 = max(0, y), min(h_img, y + h)
        x1, x2 = max(0, x), min(w_img, x + w)

        crop_img = original_img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        if crop_img.size > 0 and crop_mask.size > 0:
            return crop_img, crop_mask
    return None, None # Failed


def paste_single_product_to_shelf(product_img, shelf_img):
    """
    Blends a 448x448 product into a 1024x512 shelf.
    """
    # Setup dimensions
    s_h, s_w = shelf_img.shape[:2]  # 512, 1024
    p_h, p_w = product_img.shape[:2]  # 448, 448

    # Pick a random 448x448 crop from the shelf
    start_y = random.randint(0, s_h - p_h)
    start_x = random.randint(0, s_w - p_w)
    shelf_crop = shelf_img[start_y:start_y + p_h, start_x:start_x + p_w].copy()

    # Create the Alpha Mask from the product's black background
    gray_prod = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_prod, 1, 255, cv2.THRESH_BINARY)

    # Soften the edges (Gaussian Blur)
    kernel = np.ones((3, 3), np.uint8)
    # Erode the mask
    eroded_mask = cv2.erode(mask, kernel, iterations=3)

    alpha = eroded_mask.astype(float) / 255.0
    # Apply a smaller blur for a sharper
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha_3ch = cv2.merge([alpha, alpha, alpha])

    # Perform Alpha Blending (Westerski et al., 2024 style)
    foreground = product_img.astype(float)
    background = shelf_crop.astype(float)

    # Formula: (Product * Alpha) + (Shelf * (1 - Alpha))
    blended = (foreground * alpha_3ch) + (background * (1.0 - alpha_3ch))
    return blended.astype(np.uint8)


def paste_product_to_shelf(shelf_rgb, shelf_mask, product_rgb, product_mask, x_pos, y_pos):
    """
    Pastes a product onto the shelf.
    """
    # --- RESIZE LOGIC ---
    h, w = product_rgb.shape[:2]
    MAX_SIZE = 100
    scale_factor = MAX_SIZE / max(h, w)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    product_rgb_small = cv2.resize(product_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Use Binary Thresholding after resize to ensure clean 0/255 mask
    product_mask_small = cv2.resize(product_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    _, product_mask_small = cv2.threshold(product_mask_small, 127, 255, cv2.THRESH_BINARY)

    # --- CALCULATE COORDINATES ---
    canvas_h, canvas_w = shelf_rgb.shape[:2]
    y1 = y_pos
    y2 = min(y_pos + new_h, canvas_h)
    x1 = x_pos
    x2 = min(x_pos + new_w, canvas_w)

    paste_h = y2 - y1
    paste_w = x2 - x1

    if paste_h <= 0 or paste_w <= 0:
        return shelf_rgb, shelf_mask, None

    # Crop the source to fit the canvas bounds
    src_rgb = product_rgb_small[:paste_h, :paste_w]
    src_mask_visual = product_mask_small[:paste_h, :paste_w]

    # --- PASTE VISUALS (RGB) ---
    # Alpha Blending (Westerski et al., 2024)
    roi_rgb = shelf_rgb[y1:y2, x1:x2]

    # Create Alpha MAsk
    alpha = src_mask_visual.astype(float) / 255.0

    # Blur the Alpha Mask
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    # Stack to 3 channels (to match RGB)
    alpha_3ch = cv2.merge([alpha, alpha, alpha])

    # Perform alpha blending
    foreground = src_rgb.astype(float)
    background = roi_rgb.astype(float)

    blended = (foreground * alpha_3ch + background * (1.0 - alpha_3ch))

    # Place blended results together
    shelf_rgb[y1:y2, x1:x2] = blended.astype(np.uint8)

    # --- PASTE MASK (The 'Global Moat' Fix) ---

    # Create a temporary full-size canvas
    object_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    object_canvas[y1:y2, x1:x2] = src_mask_visual

    # Define Kernel
    kernel = np.ones((3,3), np.uint8)

    # Create the MOAT (The Eraser) on the FULL canvas
    moat_full = cv2.dilate(object_canvas, kernel, iterations=4)

    # Create the ISLAND (The Label) on the FULL canvas
    # This shrinks the object slightly
    island_full = object_canvas

    # CUT THE MOAT (Force Black)
    shelf_mask = cv2.bitwise_and(shelf_mask, shelf_mask, mask=cv2.bitwise_not(moat_full))

    # PASTE THE ISLAND (Force White)
    shelf_mask = cv2.bitwise_or(shelf_mask, island_full)

    # --- FINISH ---
    new_bbox = [x1, y1, paste_w, paste_h]
    return shelf_rgb, shelf_mask, new_bbox