import tensorflow as tf
from keras.src.optimizers.adamw import AdamW
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from model_tester.test_model import test_model

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

# optimizer = Adam(learning_rate=1e-4)
# optimizer = AdamW()
early_stopping = EarlyStopping(
    monitor='val_f1',  # specify the F1 score for early stopping
    patience=4,
    mode='max',  #
    restore_best_weights=True
)

# test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=optimizer, early_stopping=early_stopping)
# dst_path = "/Users/sosen/UniProjects/eng-thesis/data/AC8-combined/CROPPED/NON-STED"
# save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adamw/combined"
#
# test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=AdamW(), early_stopping=early_stopping,
#            attempt_name="efficientnetv2b0", finetune=True, finetune_optimizer=AdamW(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-non-sted"

test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b0")
test_model(save_path, dst_path, "efficientnetv2b1", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b1")
test_model(save_path, dst_path, "efficientnetv2b2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b2")
test_model(save_path, dst_path, "efficientnetv2b3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b3")
test_model(save_path, dst_path, "efficientnetv2s", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2s")
test_model(save_path, dst_path, "efficientnetv2m", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2m")
test_model(save_path, dst_path, "efficientnetv2l", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2l")
test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet50v2")
test_model(save_path, dst_path, "resnet101v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet101v2")
test_model(save_path, dst_path, "resnet152v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet152v2")
test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet121")
test_model(save_path, dst_path, "densenet169", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet169")
test_model(save_path, dst_path, "densenet201", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet201")
test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv2")
test_model(save_path, dst_path, "mobilenetv3small", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3small")
test_model(save_path, dst_path, "mobilenetv3large", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3large")
test_model(save_path, dst_path, "vgg16", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg16")
test_model(save_path, dst_path, "vgg19", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg19")
dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-sted"

test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b0")
test_model(save_path, dst_path, "efficientnetv2b1", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b1")
test_model(save_path, dst_path, "efficientnetv2b2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b2")
test_model(save_path, dst_path, "efficientnetv2b3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2b3")
test_model(save_path, dst_path, "efficientnetv2s", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2s")
test_model(save_path, dst_path, "efficientnetv2m", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2m")
test_model(save_path, dst_path, "efficientnetv2l", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="efficientnetv2l")
test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet50v2")
test_model(save_path, dst_path, "resnet101v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet101v2")
test_model(save_path, dst_path, "resnet152v2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="resnet152v2")
test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet121")
test_model(save_path, dst_path, "densenet169", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet169")
test_model(save_path, dst_path, "densenet201", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="densenet201")
test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv2")
test_model(save_path, dst_path, "mobilenetv3small", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3small")
test_model(save_path, dst_path, "mobilenetv3large", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="mobilenetv3large")
test_model(save_path, dst_path, "vgg16", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg16")
test_model(save_path, dst_path, "vgg19", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="vgg19")

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-non-sted"
test_model(save_path, dst_path, "inceptionv3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionv3")
test_model(save_path, dst_path, "inceptionresnetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionresnetv2")
test_model(save_path, dst_path, "nasnetlarge", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetlarge")
test_model(save_path, dst_path, "nasnetmobile", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetmobile")
test_model(save_path, dst_path, "xception", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="xception")

dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/STED"
save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam-sted"
test_model(save_path, dst_path, "inceptionv3", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionv3")
test_model(save_path, dst_path, "inceptionresnetv2", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="inceptionresnetv2")
test_model(save_path, dst_path, "nasnetlarge", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetlarge")
test_model(save_path, dst_path, "nasnetmobile", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="nasnetmobile")
test_model(save_path, dst_path, "xception", data_augmentation, optimizer="adam", early_stopping=early_stopping,
           attempt_name="xception")

# test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="resnet50v2", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="densenet121", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "efficientnetb0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetb0", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="mobilenetv2", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)

# dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-combined/NON-STED"
#
# test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetv2b0 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="resnet50v2  STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="densenet121 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "efficientnetb0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetb0 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="mobilenetv2 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
#
# save_path = "/Users/sosen/UniProjects/eng-thesis/experiments/basic-adam/rejected"
# dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-rejected/NON-STED"
#
# test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetv2b0", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="resnet50v2", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="densenet121", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "efficientnetb0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetb0", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="mobilenetv2", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
#
# dst_path = "/Users/sosen/UniProjects/eng-thesis/data/datasets-split/AC8-rejected/STED"
#
# test_model(save_path, dst_path, "efficientnetv2b0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetv2b0 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "resnet50v2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="resnet50v2  STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "densenet121", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="densenet121 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "efficientnetb0", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="efficientnetb0 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
# test_model(save_path, dst_path, "mobilenetv2", data_augmentation, optimizer=Adam(), early_stopping=early_stopping,
#            attempt_name="mobilenetv2 STED", finetune=True, finetune_optimizer=Adam(learning_rate=1e-5),
#            finetune_early_stopping=early_stopping)
