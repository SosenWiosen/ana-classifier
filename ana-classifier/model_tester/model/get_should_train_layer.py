import tensorflow as tf
def get_should_train_layer(model_name):
    if model_name == "resnet50" or model_name == "resnet101" or model_name == "resnet152":
        return lambda layer: True
    elif model_name == "resnet50v2" or model_name == "resnet101v2" or model_name == "resnet152v2":
        return lambda layer: True
    elif model_name == "densenet121" or model_name == "densenet169" or model_name == "densenet201":
        return lambda layer: True
    elif model_name == "efficientnetb0" or model_name == "efficientnetb1" or model_name=="efficientnetb2" or model_name=="efficientnetb3" or model_name=="efficientnetb4" or model_name=="efficientnetb5" or model_name=="efficientnetb6" or model_name=="efficientnetb7":
        return lambda layer: not isinstance(layer, tf.keras.layers.BatchNormalization)
    elif model_name == "efficientnetv2b0" or model_name == "efficientnetv2b1" or model_name == "efficientnetv2b2" or model_name == "efficientnetv2b3" or model_name == "efficientnetv2s" or model_name == "efficientnetv2m" or model_name == "efficientnetv2l":
        return lambda layer: not isinstance(layer, tf.keras.layers.BatchNormalization)
    elif model_name == "mobilenetv2":
        return lambda layer: True
    elif model_name == "mobilenetv3small" or model_name == "mobilenetv3large":
        return lambda layer: True
    elif model_name == "inceptionv3":
        return lambda layer: True
    elif model_name == "inceptionresnetv2":
        return lambda layer: True
    elif model_name == "xception":
        return lambda layer: True
    elif model_name == "vgg16":
        return lambda layer: True
    elif model_name == "vgg19":
        return lambda layer: True
    elif model_name == "nasnetlarge" or model_name == "nasnetmobile":
        return lambda layer: True

