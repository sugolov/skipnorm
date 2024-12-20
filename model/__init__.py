import torch 
from vit import VisionTransformer

def get_model(model_name, image_size, num_classes, **kwargs):
    if model_name == "ViT_T16":
        return VisionTransformer(
            image_size=image_size, 
            patch_size=2, 
            num_classes=num_classes, 
            dim=128, 
            depth=4, 
            heads=8, 
            mlp_dim=512, 
            channels=3, 
            dim_head=128,
            **kwargs
        )
    elif model_name == "ViT_S16":
        return VisionTransformer(
            image_size=image_size, 
            patch_size=2, 
            num_classes=num_classes, 
            dim=512, 
            depth=8, 
            heads=8, 
            mlp_dim=2048, 
            channels=3, 
            dim_head=512,
            **kwargs
        )
    elif model_name == "ViT_B16":
        return VisionTransformer(
            image_size=image_size, 
            patch_size=2, 
            num_classes=num_classes, 
            dim=768, 
            depth=12, 
            heads=12, 
            mlp_dim=3072, 
            channels=3, 
            dim_head = 768,
            **kwargs
        )
    elif model_name == "ViT_L16":
        return VisionTransformer(
            image_size=image_size, 
            patch_size=2, 
            num_classes=num_classes, 
            dim=1024, 
            depth=24, 
            heads=16, 
            mlp_dim=4096, 
            channels=3, 
            dim_head=1024,
            **kwargs
        )
    elif model_name == "ViT_XL16":
        return VisionTransformer(
            image_size=image_size, 
            patch_size=2, 
            num_classes=num_classes, 
            dim=1280, 
            depth=32, 
            heads=16, 
            mlp_dim=5120, 
            channels=3, 
            dim_head=1280,
            **kwargs
        )