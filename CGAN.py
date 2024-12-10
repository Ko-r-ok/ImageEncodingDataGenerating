import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from torchinfo import summary


class CGAN(nn.Module):
    def __init__(self, batch_size, d_in_img_s, d_out_img_s, g_mid_img_s, n_classes, conditional=True, greyscale=False):
        super(CGAN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.noise_size = 100
        self.d_in_img_s = d_in_img_s
        self.greyscale = greyscale

        # init the networks and apply weight init
        if conditional:
            self.generator = Conditional_Generator(self.noise_size, n_classes, g_mid_img_s, greyscale)
            self.generator.apply(self.weights_init)
            self.discriminator = Conditional_Discriminator(d_in_img_s, d_out_img_s, n_classes, greyscale)
            self.discriminator.apply(self.weights_init)
        else:
            self.generator = Generator(self.noise_size, g_mid_img_s, greyscale)
            self.discriminator = Discriminator(d_in_img_s, d_out_img_s, greyscale)

        self.criterion = nn.BCELoss() # the loss is BCELoss for both the D and G
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def get_labels(self, real=True):
        # real labels are 1 and fake labels 0
        if real:
            return torch.ones(self.batch_size, 1, device=self.device)
        else:
            return torch.zeros(self.batch_size, 1, device=self.device)

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

    # sample 'amount' amount of data from each class, reverse the standard scaling and rounds up
    def sample(self, amount, original_data, autoencoder, scaler, round_exceptions):
        self.set_mode("eval")
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

        # round the necessary columns -- everything except for the exceptions
        fake_samples.loc[:, ~fake_samples.columns.isin(round_exceptions)] = fake_samples.loc[:, ~fake_samples.columns.isin(round_exceptions)].round(0)

        # set the column types equal to the ones in the original
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
        return self.discriminator(x, labels + torch.rand(self.batch_size, device=self.device) * 0.00001)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_summary(self):
        print("Summary of the Generator")
        summary(self.generator, input_size=[(1, 100), (1,)])
        print("")
        print("Summary of the Discriminator")
        summary(self.discriminator, input_size=[(1, 1 if self.greyscale else 3, self.d_in_img_s, self.d_in_img_s), (1,)])

    def opt_g(self, fake_result):
        g_loss = self.criterion(fake_result, self.get_labels(real=True))
        self.generator.zero_grad()
        g_loss.backward(retain_graph=True)
        self.g_optimizer.step()

        return g_loss.item()

    def opt_d(self, real_result, fake_result):
        loss_real = self.criterion(real_result, self.get_labels(real=True))
        loss_fake = self.criterion(fake_result, self.get_labels(real=False))
        d_loss = (loss_real + loss_fake) / 2
        self.discriminator.zero_grad()
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()

        return d_loss.item()

    def train_model(self, autoencoder, dataloader, epochs=1000):
        autoencoder.eval()
        self.set_mode("train")
        total_g_loss = []
        total_d_loss = []
        for _ in tqdm(range(epochs), colour='magenta'):
            for features, real_labels in dataloader:
                batch_size = len(features)
                self.set_batch_size(batch_size)
                real_images = autoencoder.encode(features, real_labels)

                fake_labels = torch.randint(0, self.n_classes, (self.batch_size,), device=self.device)
                fake_images = self.generate(fake_labels)

                real_result = self.discriminate(real_images, real_labels)
                fake_result = self.discriminate(fake_images, fake_labels)

                # optimizing the discriminator
                total_d_loss.append(self.opt_d(real_result, fake_result))
                # optimizing the generator
                total_g_loss.append(self.opt_g(fake_result))

        return total_d_loss, total_g_loss

class Conditional_Generator(nn.Module):
    def __init__(self, noise_size, n_classes, g_mid_img_s, greyscale):
        super(Conditional_Generator, self).__init__()
        self.g_mid_img_s = g_mid_img_s
        self.out_channels = 1 if greyscale else 3

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, n_classes),
        )

        self.generator1 = nn.Sequential(
            nn.Linear(noise_size+n_classes, 128 * self.g_mid_img_s * self.g_mid_img_s),
        )

        self.generator2 = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh()

            # the output of the generator is an RGB image if input=7x7 => output=28x28 | if input=16x16 => output=64x64
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding(labels) # first label embedding
        x = torch.cat((x, label_embedding), dim=1) # adding the label to the noise
        x = self.generator1(x).view(-1, 128, self.g_mid_img_s, self.g_mid_img_s) # upscale to img * img * channels
        return self.generator2(x) # pass it through the convT layers


class Conditional_Discriminator(nn.Module):
    def __init__(self, in_img_s, out_img_s, n_classes, greyscale):
        super(Conditional_Discriminator, self).__init__()
        self.in_img_s = in_img_s
        self.out_img_s = out_img_s
        self.in_channels = 1 if greyscale else 3

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, n_classes),
        )

        self.discriminator1 = nn.Sequential(
            nn.Linear(n_classes, 1 * self.in_img_s * self.in_img_s),
        )

        self.discriminator2 = nn.Sequential(
            nn.Conv2d(self.in_channels+1, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(256 * self.out_img_s * self.out_img_s, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding(labels)
        label_embedding = self.discriminator1(label_embedding).view(-1, 1, self.in_img_s, self.in_img_s)
        x = torch.cat([x, label_embedding], dim=1)
        return self.discriminator2(x)

class Generator(nn.Module):
    def __init__(self, noise_size, g_mid_img_s, greyscale):
        super(Generator, self).__init__()
        self.g_mid_img_s = g_mid_img_s
        self.out_channels = 1 if greyscale else 3

        self.generator1 = nn.Sequential(
            nn.Linear(noise_size, 128 * self.g_mid_img_s * self.g_mid_img_s),
        )

        self.generator2 = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh()

            # the output of the generator is an RGB image if input=7x7 => output=28x28 | if input=16x16 => output=64x64
        )

    def forward(self, x, labels=None):
        x = self.generator1(x).view(-1, 128, self.g_mid_img_s, self.g_mid_img_s) # upscale to img * img * channels
        return self.generator2(x) # pass it through the convT layers

class Discriminator(nn.Module):
    def __init__(self, in_img_s, out_img_s, greyscale):
        super(Discriminator, self).__init__()
        self.in_img_s = in_img_s
        self.out_img_s = out_img_s
        self.in_channels = 1 if greyscale else 3

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(256 * self.out_img_s * self.out_img_s, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels=None):
        return self.discriminator2(x)