import timm

def download_model(model_name, num_classes, pretrained):
    model = timm.create_model(model_name,num_classes=num_classes, pretrained=pretrained)
    return model

