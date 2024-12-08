from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def get_base_model(model_name,input_shape, input_tensor):
    base_model = None
    if model_name == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None,
        )
    elif model_name == "resnet101":
        base_model = tf.keras.applications.ResNet101(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "resnet152":
        base_model = tf.keras.applications.ResNet152(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "resnet50v2":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "resnet101v2":
        base_model = tf.keras.applications.ResNet101V2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "resnet152v2":
        base_model = tf.keras.applications.ResNet152V2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "densenet121":
        base_model = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "densenet169":
        base_model = tf.keras.applications.DenseNet169(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "densenet201":
        base_model = tf.keras.applications.DenseNet201(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb1":
        base_model = tf.keras.applications.EfficientNetB1(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb2":
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb4":
        base_model = tf.keras.applications.EfficientNetB4(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb5":
        base_model = tf.keras.applications.EfficientNetB5(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb6":
        base_model = tf.keras.applications.EfficientNetB6(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetb7":
        base_model = tf.keras.applications.EfficientNetB7(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2b0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2b1":
        base_model = tf.keras.applications.EfficientNetV2B1(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2b2":
        base_model = tf.keras.applications.EfficientNetV2B2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2b3":
        base_model = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2s":
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2m":
        base_model = tf.keras.applications.EfficientNetV2M(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "efficientnetv2l":
        base_model = tf.keras.applications.EfficientNetV2L(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "mobilenetv3small":
        base_model = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "mobilenetv3large":
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "inceptionv3":
        base_model = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "inceptionresnetv2":
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "nasnetlarge":
        base_model = tf.keras.applications.NASNetLarge(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )
    elif model_name == "nasnetmobile":
        base_model = tf.keras.applications.NASNetMobile(
            include_top=False, weights="imagenet", input_tensor=input_tensor, input_shape=input_shape, pooling=None
        )

    return base_model
