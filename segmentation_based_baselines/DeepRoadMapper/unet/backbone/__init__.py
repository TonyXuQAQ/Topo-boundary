from . import resnet

def build_backbone(back_bone):
    if back_bone == "resnet101":
        return resnet.ResNet101(pretrained=True)