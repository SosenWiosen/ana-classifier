import base64
from io import BytesIO

import tensorflow as tf
from PIL import Image


def decode_and_preprocess_image(image_data, model_config):
    # Decode base64 image data and load it as a PIL image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert("RGB")  # Ensure 3 channels (RGB)

    # Resize image to match model input size
    target_size = (model_config["input_shape"][0], model_config["input_shape"][1])
    image = image.resize(target_size)

    # Convert to TensorFlow tensor and normalize to [0, 1]
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)  # TensorFlow tensor
    # Add batch dimension (TensorFlow expects a batch of inputs for inference)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)
    if model_config["copy_red_channel"]:
        red_channel = image_tensor[..., 0:1]  # Extract only the red channel, shape (H, W, 1)
        image_tensor = tf.concat([red_channel, red_channel, red_channel], axis=-1)
    return image_tensor
