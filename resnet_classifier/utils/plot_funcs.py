import pylab as plt
import numpy as np

def plot_images(images,labels,classes):
    img_len = len(labels)
    figure, ax = plt.subplots(1,img_len)
    for i, image, label in zip(range(img_len), images, labels):
        ax[i].imshow(np.transpose(image, (1, 2, 0)))
        ax[i].set_xlabel(classes[label])
    plt.show()




