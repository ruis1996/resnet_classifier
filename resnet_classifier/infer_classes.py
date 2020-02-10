import torch
import fire
from torchvision import transforms
# from skimage.io import imread
from PIL import Image
from glob import glob
from omegaconf import OmegaConf

import multiprocessing
import sys
import os
import time
sys.path.append("./resnet_classifier")
from resnet import resnet50


def main(cfg_p="./scripts/test.yaml"):

    omgcfg = OmegaConf.load(cfg_p)   

    src_p = omgcfg.src_p
    model_p = omgcfg.model_p
    cuda_dev = omgcfg.cuda_dev
    idx_to_class = omgcfg.idx_to_class
    
    transform = transforms.Compose(
        [transforms.Resize(512),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    device = torch.device(omgcfg.dev)

    net = resnet50(num_classes=len(idx_to_class))
    
    net.load_weights(model_p,device)
    net.to(device)

    
    # load img 
    print(src_p)
    img_p_s =  sorted(glob(src_p+ "*.jpg"))
    with torch.no_grad():
        for img_p in img_p_s:
            t_start = time.time()
            # img = imread(img_p)
            img = Image.open(img_p)
            # img = img.transpose(2, 0, 1)
            # img = torch.from_numpy(img).unsqueeze(0)
            img = transform(img).unsqueeze(0)
            img = img.to(device)
            
            outputs = net(img)
            _,labels = torch.max(outputs, 1)
            
            label = labels[0].item()
            print(f"{os.path.split(img_p)[1]}, label: {idx_to_class[label]}, time: {(time.time() - t_start):.4f}s")

            

if __name__ == "__main__":
    fire.Fire(main)
    