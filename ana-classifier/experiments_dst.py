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
dataset_tuples = [
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/CROPPED/NON-STED", "AC8-combined_CROPPED_NON-STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/CROPPED/STED", "AC8-combined_CROPPED_STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/NON-CROPPED/NON-STED", "AC8-combined_NON-CROPPED_NON-STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/NON-CROPPED/STED", "AC8-combined_NON-CROPPED_STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/brightest_squares_64/NON-STED", "AC8-combined_brightest_squares_64_NON-STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/brightest_squares_64/STED", "AC8-combined_brightest_squares_64_STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/brightest_squares_128/NON-STED", "AC8-combined_brightest_squares_128_NON-STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-combined/brightest_squares_128/STED", "AC8-combined_brightest_squares_128_STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-rejected/CROPPED/NON-STED", "AC8-rejected_CROPPED_NON-STED"),
    # ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-rejected/CROPPED/STED", "AC8-rejected_CROPPED_STED"),
    ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-rejected/NON-CROPPED/NON-STED", "AC8-rejected_NON-CROPPED_NON-STED"),
    ("/home/homelab/sosen/data/datasets-all/datasets-unsplit/AC8-rejected/NON-CROPPED/STED", "AC8-rejected_NON-CROPPED_STED"),
    ("/home/homelab/sosen/data/datasets-all/i3a", "i3a"),
]

# Parameters shared across all models
common_params = {
    "max_epochs": 50,
    "finetune": True,
    "finetune_max_epochs": 100,
}

for dst_path, dst_name in dataset_tuples:
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
                save_path=  "/home/homelab/sosen/experiments/dst_compared",
                dst_path=dst_path,
                attempt_name=f"{dst_name}_{attempt_name}",
                optimizer=local_optimizer,
                finetune_optimizer = local_finetune_optimizer,
                data_augmentation=data_augmentation,
                early_stopping=early_stopping,
                top= "simple",
                finetune_layers=40,
                finetune_early_stopping=finetune_early_stopping,
                **common_params)
