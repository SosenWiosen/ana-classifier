import tensorflow as tf
from keras.src.optimizers.adamw import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from model_tester.test_model import test_model



# optimizer = Adam(learning_rate=1e-4)
# optimizer = AdamW()


# Define a list of model configurations
model_configs = [
    ("efficientnetv2b0", "efficientnetv2b0"),
    ("resnet50v2", "resnet50v2"),
]

# Parameters shared across all models
common_params = {
    "save_path": "/home/homelab/sosen/experiments/basic_transfer_class_weight_augmentation",
    "dst_path": "/home/homelab/sosen/data/datasets-all/datasets-split/AC8-combined/NON-STED",
    "top": "dense1024_dropout_avg",
    "max_epochs": 30,
    "finetune": False,
    "finetune_max_epochs": 100,
    "finetune_layers": 30,
}

for model_name, attempt_name in model_configs:
    for i in range(21):  # Create 21 steps from 0 to 1 (inclusive) with increments of 0.05
        rotation_factor = i * 0.05  # Calculate current rotation factor

        # Recreate Adam optimizers for each experiment
        local_optimizer = Adam()
        local_finetune_optimizer = Adam(learning_rate=1e-5)

        # Define the data augmentation with the current rotation factor
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(rotation_factor),
        ])

        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_f1',
            patience=5,
            mode='max',
            restore_best_weights=True
        )

        # Run the test_model function with the current setup
        test_model(
            model_name=model_name,
            attempt_name=f"rotation_{rotation_factor:.2f}_{attempt_name}",
            optimizer=local_optimizer,
            data_augmentation=data_augmentation,
            early_stopping=early_stopping,
            **common_params
        )

for model_name, attempt_name in model_configs:
    # Ensure a new optimizer is created each time
    local_optimizer = Adam()
    local_finetune_optimizer = Adam(learning_rate=1e-5)

    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.RandomFlip("horizontal"),
        # tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomContrast(0.05),
        # tf.keras.layers.RandomRotation(0.5),
    ])

    early_stopping = EarlyStopping(
        monitor='val_f1',  # specify the F1 score for early stopping
        patience=5,
        mode='max',  #
        restore_best_weights=True
    )


    # Include the newly created optimizers in the call
    test_model(model_name=model_name,
               attempt_name="random_contrast_"+attempt_name,
               optimizer=local_optimizer,
               data_augmentation=data_augmentation,
               early_stopping=early_stopping,
               **common_params)

for model_name, attempt_name in model_configs:
    # Ensure a new optimizer is created each time
    local_optimizer = Adam()
    local_finetune_optimizer = Adam(learning_rate=1e-5)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        # tf.keras.layers.RandomContrast(0.05),
        # tf.keras.layers.RandomRotation(0.5),
    ])

    early_stopping = EarlyStopping(
        monitor='val_f1',  # specify the F1 score for early stopping
        patience=5,
        mode='max',  #
        restore_best_weights=True
    )


    # Include the newly created optimizers in the call
    test_model(model_name=model_name,
               attempt_name="flips_"+attempt_name,
               optimizer=local_optimizer,
               data_augmentation=data_augmentation,
               early_stopping=early_stopping,
               **common_params)
for model_name, attempt_name in model_configs:
    # Ensure a new optimizer is created each time
    local_optimizer = Adam()
    local_finetune_optimizer = Adam(learning_rate=1e-5)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomContrast(0.05),
        tf.keras.layers.RandomRotation(0.5),
    ])

    early_stopping = EarlyStopping(
        monitor='val_f1',  # specify the F1 score for early stopping
        patience=5,
        mode='max',  #
        restore_best_weights=True
    )


    # Include the newly created optimizers in the call
    test_model(model_name=model_name,
               attempt_name="all_augmentations"+attempt_name,
               optimizer=local_optimizer,
               data_augmentation=data_augmentation,
               early_stopping=early_stopping,
               **common_params)
