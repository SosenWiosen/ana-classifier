import tensorflow as tf
def get_should_train_layer(model_name):
    return lambda layer: not isinstance(layer, tf.keras.layers.BatchNormalization)

