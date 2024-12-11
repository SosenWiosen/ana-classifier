import tensorflow as tf
from keras.src.optimizers.adamw import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/avg1024-adam-non-sted"

test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b0", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2b1", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b1", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2b2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2b3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b3", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2s", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2s", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2m", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2m", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2l", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2l", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet50v2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "resnet101v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet101v2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "resnet152v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet152v2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet121", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "densenet169", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet169", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "densenet201", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet201", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "mobilenetv3small", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3small", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "mobilenetv3large", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3large", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "vgg16", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg16", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "vgg19", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg19", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/avg1024-adam-sted"

test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b0", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2b1", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b1", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2b2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2b3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b3", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2s", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2s", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2m", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2m", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "efficientnetv2l", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2l", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet50v2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "resnet101v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet101v2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "resnet152v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet152v2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet121", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "densenet169", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet169", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "densenet201", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet201", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "mobilenetv3small", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3small", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "mobilenetv3large", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3large", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "vgg16", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg16", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "vgg19", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg19", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-non-sted"
test_model(save_path, dst_path, "inceptionv3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionv3", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "inceptionresnetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionresnetv2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "nasnetlarge", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetlarge", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "nasnetmobile", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetmobile", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "xception", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="xception", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-sted"
test_model(save_path, dst_path, "inceptionv3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionv3", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "inceptionresnetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionresnetv2", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "nasnetlarge", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetlarge", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "nasnetmobile", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetmobile", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
test_model(save_path, dst_path, "xception", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="xception", max_epochs=30, finetune=True, finetune_early_stopping=early_stopping,
           finetune_max_epochs=20, finetune_optimizer=Adam(learning_rate=1e-5))
