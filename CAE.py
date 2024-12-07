import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torchinfo import summary

import Plots

class Autoencoder(nn.Module):
    def __init__(self, n_classes):
        super(Autoencoder, self).__init__()
        self.start_img_s = 7
        self.latent_img_s = 28
        self.n_classes = n_classes
        self.criterion = nn.MSELoss()

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, 10),
            nn.Linear(10, self.start_img_s * self.start_img_s),
            nn.Linear(10, self.latent_img_s * self.latent_img_s),
        )

        # Encoder network
        self.encoder1 = nn.Sequential(
            nn.Linear(8, self.start_img_s * self.start_img_s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.encoder2 = nn.Sequential(
            nn.ConvTranspose2d(2, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(7*7*128, 8),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def forward(self, x,  labels):
        return self.decode(self.encode(x, labels), labels)

    def encode(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding[0](labels)
        label_embedding = self.embedding[1](label_embedding).view(-1, 1, self.start_img_s, self.start_img_s)

        x = self.encoder1(x).view(-1, 1, self.start_img_s, self.start_img_s)
        x = torch.cat([x, label_embedding], dim=1)
        return self.encoder2(x)

    def decode(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding[0](labels)
        label_embedding = self.embedding[2](label_embedding).view(-1, 1, self.latent_img_s, self.latent_img_s)

        x = torch.cat([x, label_embedding], dim=1)
        return self.decoder(x)

    def get_summary(self):
        print("Summary of the Autoencoder")
        summary(self, input_size=[(1, 1, 1, 8), (1,)])

    def train_model(self, dataloader, epochs=50):
        self.train()
        total_loss = []
        for _ in tqdm(range(epochs), colour="yellow"):
            for inputs, labels in dataloader:
                output = self.forward(inputs, labels)
                self.optimizer.zero_grad()
                loss = self.criterion(output, inputs)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss.append(loss.item())

        # plot the loss curve after pre-training the AE
        Plots.plot_curve("Pretrained AE", total_loss)
