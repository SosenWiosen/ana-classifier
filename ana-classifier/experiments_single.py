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
    "max_epochs": 50,
    "finetune": True,
    "finetune_max_epochs": 100,
    "top": "dense1024_dropout_avg",
    "save_path": "/home/homelab/sosen/experiments/export",
    "dst_path": "/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/CROPPED/NON-STED",
}

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
            monitor='val_f1',  # specify the F1 score for ea
            patience=15,
            mode='max',  #
            restore_best_weights=True
        )
        finetune_early_stopping = EarlyStopping(
            monitor='val_f1',  # specify the F1 score for ea
            patience=15,
            mode='max',  #
            restore_best_weights=True
        )

            # Include the newly created optimizers in the call, pass the current `top`
        test_model(model_name=model_name,
                attempt_name=f"{attempt_name}",
                optimizer=local_optimizer,
                finetune_optimizer = local_finetune_optimizer,
                data_augmentation=data_augmentation,
                early_stopping=early_stopping,
                finetune_layers=40,
                finetune_early_stopping=finetune_early_stopping,
                **common_params)
