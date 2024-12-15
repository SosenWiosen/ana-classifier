import tensorflow as tf
from keras.src.optimizers.adamw import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from model_tester.test_model import test_model

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomContrast(0.05),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),
    tf.keras.layers.RandomRotation(0.1),
])

# optimizer = Adam(learning_rate=1e-4)
# optimizer = AdamW()
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

# Define a list of model configurations
model_configs = [
    ("efficientnetv2b0", "efficientnetv2b0"),
    ("efficientnetv2b1", "efficientnetv2b1"),
    ("efficientnetv2b2", "efficientnetv2b2"),
    ("efficientnetv2b3", "efficientnetv2b3"),
    ("efficientnetv2-s", "efficientnetv2-s"),
    ("efficientnetv2-m", "efficientnetv2-m"),
    ("efficientnetv2-l", "efficientnetv2-l"),
    ("resnet50v2", "resnet50v2"),
    ("resnet101v2", "resnet101v2"),
    ("resnet152v2", "resnet152v2"),
    ("densenet121", "densenet121"),
    ("densenet169", "densenet169"),
    ("densenet201", "densenet201"),
    ("mobilenetv2", "mobilenetv2"),
    ("mobilenetv3small", "mobilenetv3small"),
    ("mobilenetv3large", "mobilenetv3large"),
    ("vgg16", "vgg16"),
    ("vgg19", "vgg19"),
    ("inceptionv3", "inceptionv3"),
    ("inceptionresnetv2", "inceptionresnetv2"),
    ("nasnetlarge", "nasnetlarge"),
    ("nasnetmobile", "nasnetmobile"),
    ("xception", "xception")
]

# Parameters shared across all models
common_params = {
    "save_path": "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-non-sted",
    "dst_path": "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED",
    "data_augmentation": data_augmentation,  # Assuming this is defined elsewhere
    "top": "dense1024_dropout_avg",
    "optimizer": Adam(learning_rate=1e-4),
    "max_epochs": 80,
    "early_stopping": early_stopping,  # Assuming this is defined elsewhere
    "finetune": True,
    "finetune_optimizer": Adam(learning_rate=1e-5),
    "finetune_early_stopping": finetune_early_stopping,  # Assuming this is defined elsewhere
    "finetune_max_epochs": 100,
    "finetune_layers": 30
}

for model_name, attempt_name in model_configs:
    test_model(model=model_name,
               attempt_name=attempt_name,
               **common_params)  # Unpacking common parameters


common_params = {
    "save_path": "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-sted",
    "dst_path": "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED",
    "data_augmentation": data_augmentation,  # Assuming this is defined elsewhere
    "top": "dense1024_dropout_avg",
    "optimizer": Adam(learning_rate=1e-4),
    "max_epochs": 80,
    "early_stopping": early_stopping,  # Assuming this is defined elsewhere
    "finetune": True,
    "finetune_optimizer": Adam(learning_rate=1e-5),
    "finetune_early_stopping": finetune_early_stopping,  # Assuming this is defined elsewhere
    "finetune_max_epochs": 100,
    "finetune_layers": 30
}

for model_name, attempt_name in model_configs:
    test_model(model=model_name,
               attempt_name=attempt_name,
               **common_params)  # Unpacking common parameters
