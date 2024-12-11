import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import GAN
import Support


batch_size= 128
n_classes = 10
device = Support.print_default()

use_pretrained = False
conditional = False

gan = GAN.GAN(batch_size, 28, 7, 7, n_classes, conditional=conditional, greyscale=True).to(device)

if use_pretrained:
    gan.get_generator().load_state_dict(torch.load('weights/mnist_cgen.pth' if conditional else 'weights/mnist_gen.pth'))
    gan.get_discriminator().load_state_dict(torch.load('weights/mnist_cdis.pth' if conditional else 'weights/mnist_dis.pth'))
else:
    gan.set_mode("train")

    #printing the summary of the networks
    gan.get_summary()
    def load_data():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1 - x),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
        print(f"Len of the dataset: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Len of the dataloader: {len(loader)}")
        return loader

    dataloader = load_data()

    # train the gan
    total_d_loss = []
    total_g_loss = []
    for epoch in tqdm(range(10)):
        for i, (real_images, real_labels) in enumerate(dataloader):
            batch_size = real_images.size(0)
            gan.set_batch_size(batch_size)
            real_images, real_labels = real_images.to(device), real_labels.to(device)

            fake_labels = torch.randint(0, n_classes, (batch_size,), device=device)
            fake_images = gan.generate(fake_labels)

            real_result = gan.discriminate(real_images, real_labels)
            fake_result = gan.discriminate(fake_images, fake_labels)

            total_g_loss.append(gan.opt_g(fake_result))
            total_d_loss.append(gan.opt_d(real_result, fake_result))


    # save the weights
    torch.save(gan.get_generator().state_dict(), "weights/mnist_cgen.pth" if conditional else "weights/mnist_gen.pth")
    torch.save(gan.get_discriminator().state_dict(), "weights/mnist_cdis.pth" if conditional else "weights/mnist_dis.pth")

    Support.plot_curve("Discriminator Loss", total_d_loss)
    Support.plot_curve("Generator Loss", total_g_loss)

# printing out 10 of each clothes
gan.set_mode("eval")
examples_per_class = 10
with torch.no_grad():
    # Generate class labels
    gan.set_batch_size(n_classes * examples_per_class)
    class_labels = torch.arange(n_classes, device=device).repeat_interleave(examples_per_class)
    # Generate fake images
    generated_images = gan.generate(class_labels)

# Plot the gigaplot
Support.plot_gray_gigaplot(generated_images, "Generated Images MNIST", n_classes, examples_per_class)