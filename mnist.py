import random
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import CGAN
import Plots

device = Plots.print_default()
BATCH_SIZE = 100
n_classes = 10

cgan = CGAN.CGAN(BATCH_SIZE, 28, 7, 7, n_classes, greyscale=True).to(device)
cgan.set_mode("train")

#printing the summary of the networks
cgan.get_summary()

# reading the dataset and transforming it
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x), # flip the colours
        transforms.Normalize((0.5,), (0.5,))
    ])
dataset = datasets.FashionMNIST('.', train=True, transform=transform, download=True)
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i in range(10):
    image, label = dataset[i]
    # Convert tensor to numpy array and remove the channel dimension
    image = image.squeeze().numpy()
    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# train the gan
total_d_loss = []
total_g_loss = []
for epoch in tqdm(range(50)):
    for i, (real_images, real_labels) in enumerate(dataloader):

        real_images, real_labels = real_images.to(device), real_labels.to(device)

        fake_labels = torch.randint(0, n_classes, (BATCH_SIZE,), device=device)
        fake_images = cgan.generate(fake_labels)

        real_result = cgan.discriminate(real_images, real_labels)
        fake_result = cgan.discriminate(fake_images, fake_labels)

        total_d_loss.append(cgan.opt_d(real_result, fake_result))
        total_g_loss.append(cgan.opt_g(fake_result))

# save the weights
torch.save(cgan.get_generator().state_dict(), "mnist_gen.pt")
torch.save(cgan.get_discriminator().state_dict(), "mnist_dis.pt")

Plots.plot_curve("Discriminator Loss", total_d_loss)
Plots.plot_curve("Generator Loss", total_g_loss)

# printing out 10 of each clothes
cgan.set_mode("eval")
examples_per_class = 10
with torch.no_grad():
    # Generate class labels
    cgan.set_batch_size(n_classes * examples_per_class)
    class_labels = torch.arange(n_classes, device=device).repeat_interleave(examples_per_class)
    # Generate fake images
    generated_images = cgan.generate(class_labels)

# Plot the gigaplot
Plots.plot_gray_gigaplot(generated_images, n_classes, examples_per_class)