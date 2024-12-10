import tensorflow as tf
from keras.src.optimizers.adamw import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from model_tester.test_model import test_model


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.05),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),
])

# optimizer = Adam(learning_rate=1e-4)
# optimizer = AdamW()
early_stopping = EarlyStopping(
    monitor='val_f1',  # specify the F1 score for early stopping
    patience=4,
    mode='max',  #
    restore_best_weights=True
)



dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-non-sted"
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=AdamW(), early_stopping=early_stopping,
           attempt_name="efficientnetv2b0", finetune=True, finetune_optimizer=AdamW(learning_rate=1e-5),
           finetune_early_stopping=early_stopping)
dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-sted"

