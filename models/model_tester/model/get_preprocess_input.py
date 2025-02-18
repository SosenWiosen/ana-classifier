import tensorflow as tf
def get_preprocess_input(model_name):
    if model_name == "resnet50" or model_name == "resnet101" or model_name == "resnet152":
        return tf.keras.applications.resnet.preprocess_input
    elif model_name == "resnet50v2" or model_name == "resnet101v2" or model_name == "resnet152v2":
        return tf.keras.applications.resnet_v2.preprocess_input
    elif model_name == "densenet121" or model_name == "densenet169" or model_name == "densenet201":
        return tf.keras.applications.densenet.preprocess_input
    elif model_name == "efficientnetb0" or model_name == "efficientnetb1" or model_name=="efficientnetb2" or model_name=="efficientnetb3" or model_name=="efficientnetb4" or model_name=="efficientnetb5" or model_name=="efficientnetb6" or model_name=="efficientnetb7":
        return tf.keras.applications.efficientnet.preprocess_input
    elif model_name == "efficientnetv2b0" or model_name == "efficientnetv2b1" or model_name == "efficientnetv2b2" or model_name == "efficientnetv2b3" or model_name == "efficientnetv2s" or model_name == "efficientnetv2m" or model_name == "efficientnetv2l":
        return tf.keras.applications.efficientnet_v2.preprocess_input
    elif model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == "mobilenetv3small" or model_name == "mobilenetv3large":
        return tf.keras.applications.mobilenet_v3.preprocess_input
    elif model_name == "inceptionv3":
        return tf.keras.applications.inception_v3.preprocess_input
    elif model_name == "inceptionresnetv2":
        return tf.keras.applications.inception_resnet_v2.preprocess_input
    elif model_name == "xception":
        return tf.keras.applications.xception.preprocess_input
    elif model_name == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    elif model_name == "vgg19":
        return tf.keras.applications.vgg19.preprocess_input
    elif model_name == "nasnetlarge" or model_name == "nasnetmobile":
        return tf.keras.applications.nasnet.preprocess_input
