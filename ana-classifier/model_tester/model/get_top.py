from tensorflow.keras import layers
import tensorflow as tf


def get_top(base_model_output, num_classes, top, dropout_rate):
    x = None
    # Check if a string (identifier for a predefined architecture) was passed
    if isinstance(top, str):
        if top == "avgpool":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model_output)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        elif top == "maxpool":
            x = layers.GlobalMaxPooling2D(name="max_pool")(base_model_output)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        elif top == "flatten_dense":
            x = layers.Flatten()(base_model_output)
            x = layers.Dense(1000, activation='relu')(x)
    # Check if a TensorFlow Layer or model was passed
    elif isinstance(top, (tf.keras.layers.Layer, tf.keras.Model)):
        x = top(base_model_output)

    # Append the final Dense layer with softmax activation for classification
    if x is not None:
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
        return outputs
    else:
        raise ValueError("Invalid value for 'top'. Must be either a string identifier or a TensorFlow Layer/Model.")

