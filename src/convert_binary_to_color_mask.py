import os

import numpy as np
from PIL import Image, UnidentifiedImageError


def convert_binary_masks(
    input_folder, output_folder, wire_color=(255, 221, 51)
):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    mask_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    mask_files = sorted(mask_files)

    for mask_file in mask_files:
        mask_path = os.path.join(input_folder, mask_file)

        try:
            mask = Image.open(mask_path).convert("RGB")  # Convert to RGB
        except UnidentifiedImageError:
            print(f"Skipping {mask_file} due to loading error.")
            continue

        mask_np = np.array(mask)

        # Create a mask for white pixels
        white_pixels = (mask_np == [255, 255, 255]).all(axis=-1)

        # Set white pixels to the wire color
        mask_np[white_pixels] = wire_color

        # Convert back to image and save
        new_mask = Image.fromarray(mask_np)
        new_mask.save(os.path.join(output_folder, mask_file))
        print(f"Saved converted mask: {mask_file}")


# Example usage
input_folder = "data/De-fencing/dataset/Test Set/squared_labels"
output_folder = "data/De-fencing/dataset/Test Set/converted_masks"
convert_binary_masks(input_folder, output_folder, wire_color=(51, 221, 255))
