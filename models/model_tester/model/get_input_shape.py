def get_input_shape(model_name):
    if model_name == "resnet50":
        return (224, 224, 3)
    elif model_name == "resnet101":
        return (224, 224, 3)
    elif model_name == "resnet152":
        return (224, 224, 3)
    elif model_name == "resnet50v2":
        return (224, 224, 3)
    elif model_name == "resnet101v2":
        return (224, 224, 3)
    elif model_name == "resnet152v2":
        return (224, 224, 3)
    elif model_name == "densenet121":
        return (224, 224, 3)
    elif model_name == "densenet169":
        return (224, 224, 3)
    elif model_name == "densenet201":
        return (224, 224, 3)
    elif model_name == "efficientnetb0":
        return (224, 224, 3)
    elif model_name == "efficientnetb1":
        return (240, 240, 3)
    elif model_name == "efficientnetb2":
        return (260, 260, 3)
    elif model_name == "efficientnetb3":
        return (300, 300, 3)
    elif model_name == "efficientnetb4":
        return (380, 380, 3)
    elif model_name == "efficientnetb5":
        return (456, 456, 3)
    elif model_name == "efficientnetb6":
        return (528, 528, 3)
    elif model_name == "efficientnetb7":
        return (600, 600, 3)
    elif model_name == "efficientnetv2b0":
        return (260, 260, 3)
    elif model_name == "efficientnetv2b1":
        return (260, 260, 3)
    elif model_name == "efficientnetv2b2":
        return (260, 260, 3)
    elif model_name == "efficientnetv2b3":
        return (260, 260, 3)
    elif model_name == "efficientnetv2s":
        return (384, 384, 3)
    elif model_name == "efficientnetv2m":
        return (480, 480, 3)
    elif model_name == "efficientnetv2l":
        return (480, 480, 3)
    elif model_name == "mobilenetv2":
        return (224, 224, 3)
    elif model_name == "mobilenetv3small":
        return (224, 224, 3)
    elif model_name == "mobilenetv3large":
        return (224, 224, 3)
    elif model_name == "inceptionv3":
        return (299, 299, 3)
    elif model_name == "inceptionresnetv2":
        return (299, 299, 3)
    elif model_name == "xception":
        return (299, 299, 3)
    elif model_name == "vgg16":
        return (224, 224, 3)
    elif model_name == "vgg19":
        return (224, 224, 3)
    elif model_name == "nasnetlarge":
        return (331, 331, 3)
    elif model_name == "nasnetmobile":
        return (224, 224, 3)
