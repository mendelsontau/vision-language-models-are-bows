import os
from PIL import Image
from torchvision import transforms
from .constants import CACHE_DIR
import torch


def get_model(model_name, device, root_dir=CACHE_DIR):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        import clip
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "blip-flickr-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-flickr-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-coco-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-coco-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "xvlm-flickr":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-flickr")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-coco":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-coco")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "flava":
        from .flava import FlavaWrapper
        flava_model = FlavaWrapper(root_dir=root_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif model_name == "NegCLIP":
        import open_clip
        from .clip_models import CLIPWrapper
        
        path = os.path.join(root_dir, "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "coca":
        import open_clip
        from .clip_models import CLIPWrapper
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name="coca_ViT-B-32", pretrained="laion2B-s13B-b90k", device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    elif "laion-clip" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name=variant, pretrained="laion2b_s34b_b79k", device=device)
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
    elif "open_clip" in model_name:
        import sys
        sys.path.insert(0, "/home/gamir/DER-Roei/alon/open_clip_ref/src")
        from .open_clip_models import CLIPWrapper
        from open_clip import create_model_and_transforms
        import json
        f = open("configs/open_clip.json")
        open_clip_config = json.load(f)
        model, _, image_preprocess = create_model_and_transforms(
            open_clip_config["model"],
            open_clip_config["pretrained"],
            open_clip_config["lora"],
            open_clip_config["image_lora"],
            open_clip_config["text_lora"],
            open_clip_config["prompt_tokens"],
            open_clip_config["prompt_attention"],
            open_clip_config["prompt_attention_full"],
            open_clip_config["mask_attention"],
            device=device,)

        if open_clip_config["checkpoint"] != "":
            checkpoint = torch.load(open_clip_config["checkpoint"], map_location='cpu')
            if 'epoch' in checkpoint:
                sd = checkpoint["state_dict"]
                if next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)

        
        clip_model = CLIPWrapper(model, device)
        return clip_model, image_preprocess

    elif "blip" in model_name:
        from .blip_models_ours import BLIPModelWrapper


        blip_model = BLIPModelWrapper(device=device)
        image_preprocess = transforms.Compose([
                        transforms.Resize((224, 224),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
        
    else:
        raise ValueError(f"Unknown model {model_name}")
