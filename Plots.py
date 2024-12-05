from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_rgb_gigaplot(title, images, num_classes, examples_per_class):
    # Scale from [-1, 1] to [0, 255]
    images = ((images + 1) * 127.5).clip(0, 255).to(torch.uint8)

    fig, axes = plt.subplots(num_classes, examples_per_class, figsize=(examples_per_class, num_classes))
    for i in range(num_classes):
        for j in range(examples_per_class):
            idx = i * examples_per_class + j
            img = images[idx].detach().cpu().permute(1, 2, 0).numpy()  # Convert to numpy and rearrange channels
            axes[i, j].imshow(img)  # No cmap, since it's color
            axes[i, j].axis('off')
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{title}_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

def plot_gray_gigaplot(images, num_classes=10, examples_per_class=10):
    images = (images + 1) / 2
    fig, axes = plt.subplots(num_classes, examples_per_class, figsize=(examples_per_class, num_classes))
    for i in range(num_classes):
        for j in range(examples_per_class):
            idx = i * examples_per_class + j
            img = images[idx].detach().cpu().squeeze().numpy()  # Convert to numpy
            axes[i, j].imshow(img, cmap='gray')  # Assuming grayscale images
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_curve(title, array):
    curve = np.convolve(array, np.ones((1,)) / 1, mode='valid')
    plt.plot([j for j in range(len(curve))], curve, color='darkorange', alpha=1)
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.show()

def print_default():
    # getting and setting and displaying the used device
    try:
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(e)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")