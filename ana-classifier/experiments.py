import tensorflow as tf
from keras.src.legacy_tf_layers.normalization import BatchNormalization
from keras.src.optimizers.adamw import AdamW
from tensorflow.keras import layers, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense

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
    patience=6,
    mode='max',  #
    restore_best_weights=True
)

dropout_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalAveragePooling2D(name="avg_pool"),  # Pooling layer
    layers.BatchNormalization(),  # Normalization layer
    layers.Dense(1024, activation='relu'),  # Dense layer with ReLU activation
    layers.Dropout(0.2),  # Dropout for regularization
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])

dense1024_avg_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalAveragePooling2D(name="avg_pool"),  # Pooling layer
    layers.BatchNormalization(),  # Normalization layer
    layers.Dense(1024, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])

dense1024_512_avg_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalAveragePooling2D(name="avg_pool"),  # Pooling layer
    layers.BatchNormalization(),  # Normalization layer
    layers.Dense(1024, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(512, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])

dense1024_avg_no_batch_norm_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalAveragePooling2D(name="avg_pool"),  # Pooling layer
    layers.Dense(1024, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])

dense1024_max_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalMaxPooling2D(name="avg_pool"),  # Pooling layer
    layers.BatchNormalization(),  # Normalization layer
    layers.Dense(1024, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])

dense512_avg_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalAveragePooling2D(name="avg_pool"),  # Pooling layer
    layers.BatchNormalization(),  # Normalization layer
    layers.Dense(512, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])
dense2048_avg_lambda = lambda num_classes: tf.keras.Sequential([
    layers.GlobalAveragePooling2D(name="avg_pool"),  # Pooling layer
    layers.BatchNormalization(),  # Normalization layer
    layers.Dense(2048, activation='relu'),  # Dense layer with ReLU activation
    layers.Dense(num_classes, activation="softmax", name="pred")  # Output layer
])

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/efficientnetb0-tops/NON-STED"
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dropout_head", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dropout_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_avg_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_max", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_max_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense512_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense512_avg_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense2048_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense2048_avg_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_avg_no_batch_norm", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_avg_no_batch_norm_lambda)

test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_512_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_512_avg_lambda)

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/efficientnetb0-tops/STED"
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dropout_head", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dropout_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_avg_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_max", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_max_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense512_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense512_avg_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense2048_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense2048_avg_lambda)
test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_avg_no_batch_norm", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_avg_no_batch_norm_lambda)

test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), max_epochs=30,
           early_stopping=early_stopping,
           attempt_name="dense1024_512_avg", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
           finetune_max_epochs=20,
           finetune_early_stopping=early_stopping, top=dense1024_512_avg_lambda)
