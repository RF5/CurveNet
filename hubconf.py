dependencies = ['torch', 'numpy', 'sklearn']

import torch
import logging
from core.models.curvenet_cls import CurveNet

def curvenet_cls_pretrained_modelnet40(pretrained=True, progress=True, device='cuda'):
    r""" 
    CurveNet cls model pretrained on ModelNet40 point cloud classification dataset. 
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        device (str): device to load model onto ('cuda' or 'cpu' are common choices)

    The model takes in point clouds of shape (bs, 3, N), and returns either a logit distribution (bs, n_classes) over
    ModelNet40 classes (default behavior), or features from before the classifier head if the argument 
    `return_features=True` is passed to the forward call. In which case, features of shape (bs, 2048) will be returned. 

    Recommended to not use with more than 10k points in each point cloud, for speed and memory. 
    """    
    model = CurveNet()
    model = model.to(device).eval()
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url("https://github.com/RF5/CurveNet/releases/download/v0.1/model.t7", 
                                                map_location=device, progress=progress)
        ckpt = {k.replace('module.', ''): ckpt[k] for k in ckpt}
        model.load_state_dict(ckpt)
        model = model.to(device).eval()

    logging.info(f"[MODEL] CurveNet loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    del ckpt
    return model