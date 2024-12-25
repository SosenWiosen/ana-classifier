import tensorflow as tf
from keras.src.optimizers.adamw import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from model_tester.run_hypertuning_export_data import run_hypertuning_and_export
from model_tester.test_model import test_model

# optimizer = Adam(learning_rate=1e-4)
# optimizer = AdamW()


# Define a list of model configurations
model_configs = [
    ("efficientnetv2b0", "efficientnetv2b0"),
    # ("efficientnetv2b1", "efficientnetv2b1"),
    # ("efficientnetv2b2", "efficientnetv2b2"),
    # ("efficientnetv2b3", "efficientnetv2b3"),
    # ("efficientnetv2s", "efficientnetv2s"),
    # ("efficientnetv2m", "efficientnetv2m"),
    # ("efficientnetv2l", "efficientnetv2l"),
    # ("resnet50v2", "resnet50v2"),
    # ("resnet101v2", "resnet101v2"),
    # ("resnet152v2", "resnet152v2"),
    # ("densenet121", "densenet121"),
    # ("densenet169", "densenet169"),
    # ("densenet201", "densenet201"),
    # ("mobilenetv2", "mobilenetv2"),
    # ("mobilenetv3small", "mobilenetv3small"),
    # ("mobilenetv3large", "mobilenetv3large"),
    # ("vgg16", "vgg16"),
    # ("vgg19", "vgg19"),
    # ("inceptionv3", "inceptionv3"),
    # ("inceptionresnetv2", "inceptionresnetv2"),
    # ("nasnetlarge", "nasnetlarge"),
    # ("nasnetmobile", "nasnetmobile"),
    # ("xception", "xception")
]

# Parameters shared across all models
common_params = {
    "save_path": "/Users/sosen/UniProjects/eng-thesis/experiments/hypertune",
    "dst_path": "/Users/sosen/UniProjects/eng-thesis/data/datasets-all/datasets-split/AC8-combined/NON-STED",

    "max_epochs": 100,
    "finetune": False,
    "finetune_max_epochs": 100,
}

tune_working_dir = "/Users/sosen/UniProjects/eng-thesis/experiments/hypertune-working"
tune_project_name = "hypertune"
for model_name, attempt_name in model_configs:
    # Ensure a new optimizer is created each time

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomContrast(0.05),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical")
    ])

    early_stopping = EarlyStopping(
        monitor='val_f1',  # specify the F1 score for early stopping
        patience=15,
        mode='max',  #
        restore_best_weights=True
    )

    finetune_early_stopping = EarlyStopping(
        monitor='val_f1',  # specify the F1 score for early stopping
        patience=15,
        mode='max',  #
        restore_best_weights=True
    )

    # Include the newly created optimizers in the call
    run_hypertuning_and_export(model_name=model_name,
                               attempt_name=attempt_name,
                               data_augmentation=data_augmentation,
                               tuner_working_dir=tune_working_dir,
                               tuner_project_name=tune_project_name,

                               early_stopping=early_stopping,
                               finetune_early_stopping=finetune_early_stopping,
                               **common_params)

# common_params = {
#     "save_path": "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-sted",
#     "dst_path": "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED",
#     "top": "dense1024_dropout_avg",
#     "max_epochs": 50,
#     "finetune": True,
#     "finetune_max_epochs": 100,
#     "finetune_layers": 30
# }
#
# for model_name, attempt_name in model_configs:
#     # Ensure a new optimizer is created each time
#     local_optimizer = Adam()
#     local_finetune_optimizer = Adam(learning_rate=1e-5)
#
#     data_augmentation = tf.keras.Sequential([
#         tf.keras.layers.RandomContrast(0.05),
#         tf.keras.layers.RandomRotation(0.1),
#         tf.keras.layers.RandomFlip("horizontal"),
#         tf.keras.layers.RandomFlip("vertical")
#     ])
#
#     early_stopping = EarlyStopping(
#         monitor='val_f1',  # specify the F1 score for early stopping
#         patience=15,
#         mode='max',  #
#         restore_best_weights=True
#     )
#
#     finetune_early_stopping = EarlyStopping(
#         monitor='val_f1',  # specify the F1 score for early stopping
#         patience=15,
#         mode='max',  #
#         restore_best_weights=True
#     )
#
#     # Include the newly created optimizers in the call
#     test_model(model_name=model_name,
#                attempt_name=attempt_name,
#                optimizer=local_optimizer,
#                data_augmentation=data_augmentation,
#                finetune_optimizer=local_finetune_optimizer,
#                early_stopping=early_stopping,
#                finetune_early_stopping=finetune_early_stopping,
#                **common_params)
