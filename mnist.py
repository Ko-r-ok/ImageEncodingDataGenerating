import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import CGAN
import Plots

device = Plots.print_default()
BATCH_SIZE = 100
n_classes = 10

gan = CGAN.GAN(BATCH_SIZE, 28, 7, 7, n_classes, greyscale=True).to(device)
gan.set_mode("train")

#printing the summary of the networks
print("Summary of the Generator")
summary(gan.get_generator(), input_size=[(100,), (1,)])
print("")
print("Summary of the Discriminator")
summary(gan.get_discriminator(), input_size=[(1, 1, 28, 28), (1,)])

# reading the dataset and transforming it
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])
dataset = datasets.FashionMNIST('.', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# train the gan
total_d_loss = []
total_g_loss = []
for epoch in tqdm(range(50)):
    for i, (real_images, real_labels) in enumerate(dataloader):

        real_images, real_labels = real_images.to(device), real_labels.to(device)

        fake_labels = torch.randint(0, n_classes, (BATCH_SIZE,), device=device)
        fake_images = gan.generate(fake_labels)

        real_result = gan.discriminate(real_images, real_labels)
        fake_result = gan.discriminate(fake_images, fake_labels)

        total_d_loss.append(gan.opt_d(real_result, fake_result))
        total_g_loss.append(gan.opt_g(fake_result))

# save the weights
torch.save(gan.get_generator().state_dict(), "mnist_gen.pt")
torch.save(gan.get_discriminator().state_dict(), "mnist_dis.pt")

Plots.plot_curve("Discriminator Loss", total_d_loss)
Plots.plot_curve("Generator Loss", total_g_loss)

# printing out 10 of each clothes
gan.set_mode("eval")
with torch.no_grad():
    # Generate class labels
    class_labels = torch.arange(n_classes, device=device).repeat_interleave(10)
    # Generate fake images
    generated_images = gan.generate(class_labels)

# Plot the gigaplot
Plots.plot_gray_gigaplot(generated_images, n_classes, 10)