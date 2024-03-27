import tensorflow as tf
import wandb
from unet_architecture_local import load_images_from_directory

TEST_IMG_PATH = "data/test/Only_fence"

model = tf.keras.models.load_model('artifacts/models/test')

test_images = load_images_from_directory(
    TEST_IMG_PATH, (1024, 704), isMask=False
)

# converting images(numpy arrays) to tensors for tensorflow
testimages_tensors = tf.convert_to_tensor(test_images, dtype=tf.float32)

predictions = model.predict(testimages_tensors)

wandb.init(project="make_predictions", entity="fabio-renn")

for i in range(len(testimages_tensors)):
    # Prepare the original image, true mask, and predicted mask for logging
    original_image = testimages_tensors[i]  # Assuming this is the original, displayable image
    predicted_mask = predictions[i]  # Adjust as necessary for your model's output format
    
    # Log the original image and the predicted mask to wandb
    wandb.log({
        "original_image": wandb.Image(original_image, caption="Original Image"),
        "predicted_mask": wandb.Image(predicted_mask, caption="Predicted Mask")
    })

# Finish the wandb run
wandb.finish()