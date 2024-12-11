import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torchinfo import summary

class Autoencoder(nn.Module):
    def __init__(self, conditional, n_features, n_classes):
        super(Autoencoder, self).__init__()
        self.conditional = conditional
        self.n_features = n_features
        self.n_classes = n_classes
        if conditional:
            self.autoencoder = Conditional_Autoencoder(n_features, n_classes)
        else:
            self.autoencoder = Normal_Autoencoder(n_features, n_classes)

    def encode(self, x, labels):
        return self.autoencoder.encode(x, labels)

    def decode(self, x, labels):
        return self.autoencoder.decode(x, labels)

    def train_model(self, dataloader, epochs=50):
        return self.autoencoder.train_model(dataloader, epochs)

    def get_summary(self):
        print("Summary of the Autoencoder")
        summary(self.autoencoder, input_size=[(1, self.n_features), (1,)])

    def get_conditional(self):
        return self.conditional

class Conditional_Autoencoder(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Conditional_Autoencoder, self).__init__()
        self.start_img_s = 7
        self.latent_img_s = 28
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, 10),
        )

        # Encoder network
        self.lin_encoder = nn.Sequential(
            nn.Linear(10 + self.n_features, self.start_img_s * self.start_img_s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.conv_encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # Decoder network
        self.lin_decoder = nn.Sequential(
            nn.Linear(10, self.latent_img_s * self.latent_img_s),
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(256 * 7 * 7, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def encode(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding[0](labels)
        x = torch.cat([x, label_embedding], dim=1)

        x = self.lin_encoder(x).view(-1, 1, self.start_img_s, self.start_img_s)

        return self.conv_encoder(x)

    def decode(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding[0](labels)
        label_embedding = self.lin_decoder(label_embedding).view(-1, 1, self.latent_img_s, self.latent_img_s)

        x = torch.cat([x, label_embedding], dim=1)
        return self.conv_decoder(x)

    def forward(self, x,  labels=None):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, epochs):
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

        return total_loss


class Normal_Autoencoder(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Normal_Autoencoder, self).__init__()
        self.start_img_s = 7
        self.latent_img_s = 28
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()

        self.lin_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.start_img_s * self.start_img_s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.conv_encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(256 * 7 * 7, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def encode(self, x, labels=None):
        x = self.lin_encoder(x).view(-1, 1, self.start_img_s, self.start_img_s)
        return self.conv_encoder(x)

    def decode(self, x, labels=None):
        return self.conv_decoder(x)

    def forward(self, x, labels=None):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, epochs):
        self.train()
        total_loss = []
        for _ in tqdm(range(epochs), colour="yellow"):
            for inputs, _ in dataloader:
                output = self.forward(inputs)
                self.optimizer.zero_grad()
                loss = self.criterion(output, inputs)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss.append(loss.item())

        return total_loss