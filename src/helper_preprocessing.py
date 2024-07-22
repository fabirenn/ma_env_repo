import os

from PIL import Image, UnidentifiedImageError


def split_image_and_mask(
    image_folder, mask_folder, output_image_folder, output_mask_folder
):
    # Ensure the output directories exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    image_files = os.listdir(image_folder)
    image_files = sorted(image_files)
    mask_files = os.listdir(mask_folder)
    mask_files = sorted(mask_files)

    for image_file, mask_file in zip(image_files, mask_files):
        # Load image and mask
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        try:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
        except UnidentifiedImageError:
            print(
                f"Skipping {image_file} and {mask_file} due to loading error."
            )
            continue

        # Check image and mask sizes
        if image.size != (3000, 2000) or mask.size != (3000, 2000):
            print(
                f"Skipping {image_file} and {mask_file} due to incorrect size."
            )
            continue

        # Split the image and mask into 6 parts
        for i in range(3):  # Horizontal splits
            for j in range(2):  # Vertical splits
                left = i * 1000
                upper = j * 1000
                right = left + 1000
                lower = upper + 1000

                image_crop = image.crop((left, upper, right, lower))
                mask_crop = mask.crop((left, upper, right, lower))

                # Save the cropped images
                image_crop.save(
                    os.path.join(
                        output_image_folder,
                        f"{os.path.splitext(image_file)[0]}_{j}_{i}.png",
                    )
                )
                mask_crop.save(
                    os.path.join(
                        output_mask_folder,
                        f"{os.path.splitext(mask_file)[0]}_{j}_{i}.png",
                    )
                )


def crop_and_resize_images(
    image_folder,
    mask_folder,
    output_image_folder,
    output_mask_folder,
    target_size=1000,
):
    # Ensure the output directories exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # Get lists of files in the image and mask folders
    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    image_files = sorted(image_files)
    mask_files = [
        f
        for f in os.listdir(mask_folder)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    mask_files = sorted(mask_files)

    for image_file, mask_file in zip(image_files, mask_files):
        # Load image and mask
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        try:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
        except UnidentifiedImageError:
            print(
                f"Skipping {image_file} and {mask_file} due to loading error."
            )
            continue

        # Check image and mask sizes
        if image.size != (3264, 1840) or mask.size != (3264, 1840):
            print(
                f"Skipping {image_file} and {mask_file} due to incorrect size."
            )
            continue

        # Calculate cropping box to get the center 1840x1840 part
        left = (image.width - 1840) // 2
        upper = (image.height - 1840) // 2
        right = left + 1840
        lower = upper + 1840

        # Crop the center 1840x1840 part
        image_crop = image.crop((left, upper, right, lower))
        mask_crop = mask.crop((left, upper, right, lower))

        # Resize to 1000x1000
        image_resized = image_crop.resize(
            (target_size, target_size), Image.Resampling.LANCZOS
        )
        mask_resized = mask_crop.resize(
            (target_size, target_size), Image.Resampling.LANCZOS
        )

        # Save the resized images
        image_resized.save(
            os.path.join(
                output_image_folder,
                f"{image_file.split('.')[0]}_cropped_resized.png",
            )
        )
        mask_resized.save(
            os.path.join(
                output_mask_folder,
                f"{mask_file.split('.')[0]}_cropped_resized.png",
            )
        )

def resize_and_save_images(directory, target_size=(3000, 2000), suffix="_small"):
    # Get list of image files in the directory
    image_files = [
        f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files = sorted(image_files)

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        try:
            image = Image.open(image_path)
        except UnidentifiedImageError:
            print(f"Skipping {image_file} due to loading error.")
            continue
        
        # Resize the image
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)

        # Save the resized image with a new name
        new_file_name = f"{os.path.splitext(image_file)[0]}{suffix}{os.path.splitext(image_file)[1]}"
        image_resized.save(os.path.join(directory, new_file_name))
        print(f"Saved resized image: {new_file_name}")


# Example usage
directory = "data/segmented/mask"
#resize_and_save_images(directory, target_size=(3000, 2000), suffix="_small")


image_folder = "data/segmented/original"
mask_folder = "data/segmented/mask"
output_image_folder = "data/segmented/squared_images"
output_mask_folder = "data/segmented/squared_masks"

# split_image_and_mask(image_folder, mask_folder, output_image_folder,
# output_mask_folder)
split_image_and_mask(image_folder, mask_folder, output_image_folder, output_mask_folder)
