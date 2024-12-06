import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from torchinfo import summary


class GAN(nn.Module):
    def __init__(self, loss_fn, batch_size, d_in_img_s, d_out_img_s, g_mid_img_s, n_classes, greyscale):
        super(GAN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.noise_size = 100
        self.d_in_img_s = d_in_img_s

        # init the networks and apply weight init
        self.generator = Generator(n_classes, g_mid_img_s, greyscale)
        self.generator.apply(self.weights_init)
        self.discriminator = Discriminator(d_in_img_s, d_out_img_s, n_classes, greyscale)
        self.discriminator.apply(self.weights_init)

        self.criterion = loss_fn # the loss is CrossEntropy for both the D and G
        # /TODO keep tuning the LRs if necessary
        # /TODO maybe SGD opt
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.real_label = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
        self.fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)

    def get_labels(self, real=True, noise=False):
        # real labels are 1 and fake labels 0
        y = torch.full((self.batch_size,), 1 if real else 0, dtype=torch.float, device=self.device)
        # adding a little noise for the discriminator
        if noise: y += (torch.randn(self.batch_size, device=self.device) * 2 - 1) * 0.0001 # mapping between -1 & 1 and scale down
        return y

    def set_mode(self, mode):
        if mode == "train":
            self.generator.train()
            self.discriminator.train()
        elif mode == "eval":
            self.generator.eval()
            self.discriminator.eval()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def opt_g(self, fake_result):
        g_loss = self.criterion(fake_result.view(-1), self.real_label)
        self.generator.zero_grad()
        g_loss.backward(retain_graph=True)
        self.g_optimizer.step()

        return g_loss.item()

    def opt_d(self, real_result, fake_result):
        noise = (torch.randn(self.batch_size, device=self.device) * 2 - 1) * 0.0001
        real_target = self.real_label + noise
        fake_target = self.fake_label + noise
        loss_real = self.criterion(real_result.view(-1), real_target)
        loss_real.backward(retain_graph=True)
        loss_fake = self.criterion(fake_result.view(-1), fake_target)
        loss_fake.backward(retain_graph=True)
        self.d_optimizer.step()

        return loss_real.item() + loss_fake.item()

    # sample 'amount' amount of data from each class, reverse the standard scaling and rounds up
    def sample(self, amount, original_data, autoencoder, scaler):
        class_labels = torch.arange(self.n_classes, device=self.device).repeat_interleave(amount)
        self.set_batch_size(class_labels.size(0))
        generated_fake_images = self.generate(class_labels)
        fake_samples = np.array(autoencoder.decode(generated_fake_images, class_labels).squeeze(dim=0).detach().cpu())

        # convert the samples to dataframe and redo the normalizing
        fake_samples = pd.DataFrame(fake_samples)
        fake_samples = scaler.inverse_transform(fake_samples)

        # concat the fake samples with its labels and convert to dataframe
        class_labels = np.expand_dims(class_labels.detach().cpu().numpy(), axis=1)
        fake_samples = np.concatenate((fake_samples, class_labels), axis=1)
        fake_samples = pd.DataFrame(fake_samples, columns=original_data.columns)

        # round the necessary columns
        fake_samples.loc[:, ~fake_samples.columns.isin(['BMI', 'DiabetesPedigreeFunction'])] = fake_samples.loc[:,
                                                                                               ~fake_samples.columns.isin(
                                                                                                   ['BMI',
                                                                                                    'DiabetesPedigreeFunction'])].round(0)

        for column in fake_samples.columns:
            if column in original_data:
                fake_samples[column] = fake_samples[column].astype(original_data[column].dtype)

        # return both the converted synth data and the generated images
        return fake_samples, generated_fake_images

    def generate(self, labels):
        noise = torch.randn(self.batch_size, self.noise_size, device=self.device)
        return self.generator(noise, labels)

    def discriminate(self, x, labels):
        # adding a small noise to the labels
        return self.discriminator(x, labels + torch.randn(self.batch_size, device=self.device) * 0.0001)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_summary(self):
        print("Summary of the Generator")
        summary(self.generator, input_size=[(self.noise_size,), (1,)])
        print("")
        print("Summary of the Discriminator")
        summary(self.discriminator, input_size=[(1, 3, self.d_in_img_s, self.d_in_img_s), (1,)])

class Generator(nn.Module):
    def __init__(self, n_classes, g_mid_img_s, greyscale):
        super(Generator, self).__init__()
        self.g_mid_img_s = g_mid_img_s
        self.out_channels = 1 if greyscale else 3

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, 10),
            nn.Linear(10, self.g_mid_img_s * self.g_mid_img_s)
        )

        self.generator1 = nn.Sequential(
            nn.Linear(100, 128 * self.g_mid_img_s * self.g_mid_img_s),
            nn.LeakyReLU(0.2)
        )

        self.generator2 = nn.Sequential(
            nn.ConvTranspose2d(129, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(1024, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),

            # the output of the generator is an RGB image if input=7x7 => output=28x28 | if input=16x16 => output=64x64
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding[0](labels)
        label_embedding = self.embedding[1](label_embedding).view(-1, 1, self.g_mid_img_s, self.g_mid_img_s)

        x = self.generator1(x).view(-1, 128, self.g_mid_img_s, self.g_mid_img_s)
        x = torch.cat([x, label_embedding], dim=1)
        return self.generator2(x)


class Discriminator(nn.Module):
    def __init__(self, in_img_s, out_img_s, n_classes, greyscale):
        super(Discriminator, self).__init__()
        self.in_img_s = in_img_s
        self.out_img_s = out_img_s
        self.in_channels = 1 if greyscale else 3

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, 10),
            nn.Linear(10, 1 * self.in_img_s * self.in_img_s)
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.in_channels+1, 128, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),


            nn.Conv2d(128, 128, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),

            nn.Linear(128 * self.out_img_s * self.out_img_s, 1),
            nn.Sigmoid(),
        )
    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding[0](labels)
        label_embedding = self.embedding[1](label_embedding).view(-1, 1, self.in_img_s, self.in_img_s)

        x = torch.cat([x, label_embedding], dim=1)
        return self.discriminator(x)

