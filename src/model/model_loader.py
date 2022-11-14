from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

class ModelLoader:
    @classmethod
    def load(cls, model_type):
        if model_type == "retinanet":
            return retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights)
        else:
            raise NotImplementedError

    