import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata
from omegaconf import OmegaConf
import fire
from custom_dataset import gen_data_set
from utils.plot_funcs import plot_images
from resnet import resnet50



def main(cfg_p="./config/config.yaml"):
    # omgcfg
    omgcfg = OmegaConf.load(cfg_p)
    # dataloader
    trainset, valset,idx_to_class = gen_data_set(omgcfg.dataset_dir)
    
    trainloader = tdata.DataLoader(trainset, batch_size=omgcfg.batch_size, shuffle=True, num_workers=omgcfg.num_workers)
    valloader = tdata.DataLoader(valset, batch_size=omgcfg.batch_size, shuffle=False, num_workers=omgcfg.num_workers)


    # net
    num_classes = len(idx_to_class)
    net = resnet50(num_classes=num_classes)

    # test cuda
    device = torch.device("cuda:" + omgcfg.cuda_dev if torch.cuda.is_available() else "cpu")
    net.to(device)

    # loss
    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=omgcfg.lr, momentum=omgcfg.momentum)
    running_loss = 0.0
    for epoch in range(omgcfg.epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i%10 == 9:
                print("[%d, %5d] loss: %.3f" % (epoch +1 , i + 1, running_loss / 10))
                running_loss = 0.0

        # val test
        correct = 0
        total = 0
        accuracy = 0
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))
        if epoch  > 8:
            # val Acc
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # classes acc
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                accuracy = 100 * correct / total
                print("Accuracy of the network on the %d test images: %d %%" % (total, accuracy))
                for i in range(num_classes):
                    class_name = idx_to_class[i]
                    class_acc = 100 * class_correct[i] / class_total[i]
                    print(f"Accuracy of {class_name} : {class_acc:.2f}")
                if accuracy > omgcfg.save_acc:
                    net.save_weights(f"{omgcfg.model_p}_acc_{accuracy:.2f}_{epoch + 1}.pth")
                if accuracy > omgcfg.max_acc :
                    print(f"Train finished, all class Acc is {accuracy:.2d}%.")
                    break
if __name__ == "__main__":
    fire.Fire(main)
