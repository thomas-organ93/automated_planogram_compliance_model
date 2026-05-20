import cv2
import numpy as np
import random
import os
from pathlib import Path
from matplotlib import pyplot as plt

def _create_composite_background(texture_path, height, width):
    # Load grayscale texture
    texture = texture_path

    # Resize/Tile
    texture = cv2.resize(texture, (width, height))

    # Darken slightly
    texture = (texture * 0.7).astype(np.uint8)
    # Horizontal Beams
    beam_random_pos = random.randint(0, 200)
    for y in range(beam_random_pos, height, 200):
        cv2.line(texture, (0, y + 2), (width, y + 2), (20, 20, 20), 8)
        cv2.line(texture, (0, y), (width, y), (180, 180, 180), 8)
    return texture


def _preprocess_background(bg_image, set_height, set_width):
    """
    Converts a DTD texture (like wood grid) into a more 'shelf-like' background.
    """
    # Grayscale
    gray = cv2.imread(bg_image, cv2.IMREAD_GRAYSCALE)

    # Randomly flip the BG image
    if random.random() < 0.5:
        gray = cv2.rotate(gray, cv2.ROTATE_180)

    # Colorize to 'Supermarket Grey/Blue'
    # Create a dummy 3-channel image
    height, width = gray.shape
    metal_bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a metallic tint (B, G, R) -> slightly blue/grey
    metal_bg[:, :, 0] = gray * 0.9 + 20  # Blue
    metal_bg[:, :, 1] = gray * 0.9 + 20  # Green
    metal_bg[:, :, 2] = gray * 0.9 + 20  # Red

    # Reduce Contrast
    # High contrast grids confuse the edge detector.
    alpha = 0.6 # Contrast control (1.0 is original)
    beta = 30   # Brightness control
    metal_bg = cv2.convertScaleAbs(metal_bg, alpha=alpha, beta=beta)
    composite_bg = _create_composite_background(metal_bg, set_height, set_width)
    composite_bg = cv2.GaussianBlur(composite_bg, (5, 5), 0)
    return composite_bg

def generate_bg_shelve(set_height=512, set_width=1024):
    try:
        # Works if running as a .py script
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        # Works if running in a notebook/shell
        script_dir = Path.cwd()
        return None

    # Check valid formats
    img_dir = os.path.join(script_dir, "DTD_bg_images")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    images = []
    for root, dirs, files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith(valid_extensions):
                # Join with 'root' to get the full path including subfolders
                full_path = os.path.join(root, f)
                images.append(full_path)

    # Choose random background
    if images:
        random_image = random.choice(images)
        full_path = os.path.join(img_dir, random_image)
        gray_scale_image = _preprocess_background(full_path, set_height, set_width)
        return gray_scale_image
    else:
        print("No images found in the directory.")
        return None


def apply_camera_roll(image, instance_mask, max_angle=4):
    """
    Rotates the entire scene (Image + Mask) to simulate camera tilt.
    """
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get Rotation Matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate RGB Image (Cubic looks best)
    rotated_image = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Rotate Instance Mask
    rotated_mask = cv2.warpAffine(
        instance_mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return rotated_image, rotated_mask


def apply_camera_roll_single(image, max_angle=4):
    angle = random.uniform(-max_angle, max_angle)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Calculate the sine and cosine of the angle
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take the translation into account
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation with the new dimensions
    rotated_image = cv2.warpAffine(
        image, M, (nW, nH),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return rotated_image


if __name__ == "__main__":
    new_shelve_img = generate_bg_shelve()
    if new_shelve_img is not None:
        plt.imshow(new_shelve_img)
        cv2.imwrite("RESULT3.jpg", new_shelve_img)
        plt.tight_layout()
        plt.show()