import tensorflow as tf
import base64
from PIL import Image
from io import BytesIO

def decode_and_preprocess_image(image_data, model_config):
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = image.resize((model_config["input_shape"][0], model_config["input_shape"][1]))
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor
