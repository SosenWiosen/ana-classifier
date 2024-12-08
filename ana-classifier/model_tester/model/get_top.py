from tensorflow.python.keras import layers


def get_top(base_model_output, num_classes ,top, dropout_rate):
    outputs = None
    if top == "avgpool":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model_output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    elif top == "maxpool":
        x = layers.GlobalMaxPooling2D(name="max_pool")(base_model_output)
        x = layers.BatchNormalization()(x)

        x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    elif top == "flatten_dense":
        x = layers.Flatten()(base_model_output)
        x = layers.Dense(1000, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    return outputs
