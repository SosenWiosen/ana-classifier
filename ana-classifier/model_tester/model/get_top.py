from tensorflow.keras import layers
import tensorflow as tf


from tensorflow.keras import layers

def get_top(base_model_output, num_classes, top, dropout_rate):
    if callable(top):
        # Trust the callable to correctly handle the creation of all necessary layers
        # including whatever final layers it deems necessary.
        outputs = top(num_classes)(base_model_output)
    else:
        # Use predefined top configurations ('avgpool', 'maxpool', 'flatten_dense')
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

        # Apply the final classification only in non-callable cases
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    return outputs

