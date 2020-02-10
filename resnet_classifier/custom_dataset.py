import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

from omegaconf import OmegaConf


transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(512),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def gen_data_set(root=None):
    """
        return: trainset, valset, {idx:class}
    """
    if root is None:
        raise ValueError("must set root!")
    trainset = ImageFolder(root+"train",transform=transform)
    valset = ImageFolder(root+"val", transform=transform)
    trainset.idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}
    valset.idx_to_class = {v: k for k, v in valset.class_to_idx.items()}
    #a = wulandata.__getitem__(500)

    if trainset.idx_to_class!= valset.idx_to_class:
        raise ValueError("train and val folder net equal.")
    return trainset, valset, trainset.idx_to_class

if __name__ == "__main__":
    _, _, idx_to_class = gen_wulan_data_set()
    print(idx_to_class)