import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

from model_tester.test import test_model

save_path = "/Users/sosen/UniProjects/eng-thesis/experiments"
dst_path = "/Users/sosen/UniProjects/eng-thesis/data/AC8-combined/CROPPED/NON-STED"

data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.RandomRotation(0.1),
    # You can add more augmentations if needed
    # tf.keras.layers.RandomZoom(0.15),
    # tf.keras.layers.RandomWidth(0.2),
    # tf.keras.layers.RandomHeight(0.2),
    # tf.keras.layers.RandomShear(0.15),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),
])

optimizer = "adam"

early_stopping = EarlyStopping(
    monitor='val_f1',  # specify the F1 score for early stopping
    patience=4,
    mode='max',  #
    restore_best_weights=True
)

# test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=optimizer, early_stopping=early_stopping)
# test_model(save_path, dst_path, "efficientnetv2b1", data_augmentation, optimizer=optimizer, early_stopping=early_stopping)
test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer=optimizer, early_stopping=early_stopping)
