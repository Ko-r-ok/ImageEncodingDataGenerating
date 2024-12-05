import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from synthcity.plugins import Plugins

import CGAN
import CAE
import Plots

device = Plots.print_default()
BATCH_SIZE = 16
n_classes = 2

# reading the data
data = pd.read_csv('diabetes.csv')
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
# Convert to tensors, concat and then convert to dataloader
features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=device)
labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# setting up the networks
# /TODO maybe use some other image ganarator
autoencoder = CAE.Autoencoder(n_classes).to(device)
# TODO ctgan tvae compare
# TODO hyperparameter optimization
# TODO compare with different architecture
# TODO add the mnist result to the report
gan = CGAN.GAN(nn.CrossEntropyLoss(), BATCH_SIZE, 28, 5, 7, 2, False).to(device)
gan.set_mode("train")

# print the summary of the AE and G
autoencoder.get_summary()
print("")
gan.get_summary()

# pre-train the encoder
total_ae_loss = []
autoencoder.train()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)
for _ in tqdm(range(50), colour="yellow"):
    for inputs, labels in dataloader:
        output = autoencoder(inputs, labels)
        optimizer.zero_grad()
        loss = criterion(output, inputs)
        loss.backward()
        optimizer.step()
        total_ae_loss.append(loss.item())

# plot the loss curve after pre-training the AE
Plots.plot_curve("Pretrained AE", total_ae_loss)

# gan training
autoencoder.eval()
total_g_loss = []
total_d_loss = []
for _ in tqdm(range(1000), colour='magenta'):
    for features, real_labels in dataloader:
        real_images = autoencoder.encode(features, real_labels)

        fake_labels = torch.randint(0, n_classes, (BATCH_SIZE,), device=device)
        fake_images = gan.generate(fake_labels)

        real_result = gan.discriminate(real_images, real_labels)
        fake_result = gan.discriminate(fake_images, fake_labels)

        # optimizing the discriminator
        total_d_loss.append(gan.opt_d(real_result, fake_result))

        # optimizing the generator
        total_g_loss.append(gan.opt_g(fake_result))


# plot the loss curve for the G and D
Plots.plot_curve("Generator", total_g_loss)
Plots.plot_curve("Discriminator", total_d_loss)

# generate 2x5 labels and 2x5 fake samples
gan.set_mode("eval")
fake_samples, generated_fake_images = gan.sample(5, data.head(5), autoencoder, scaler)
# display 10 fake data sample
print("10 fake samples:")
print(fake_samples.to_string())

# Plot the 10 fake images in  a giga-plot
Plots.plot_rgb_gigaplot("GAN", generated_fake_images, n_classes, 5)

# Select first 5 elements where label is 0 and the first 5 where label is 1
class_0 = data[data['Outcome'] == 0].iloc[:5]
class_1 = data[data['Outcome'] == 1].iloc[:5]
combined = pd.concat([class_0, class_1])

# plot that 5-5 real samples
print("")
print("5 real samples:")
print(combined.head(10).to_string())

labels = combined.iloc[:, [-1]]
combined = combined.drop(columns=['Outcome'])
combined = torch.tensor(combined.values, dtype=torch.float32, device=device).unsqueeze(0)
labels = torch.tensor(labels.values, dtype=torch.float32, device=device).unsqueeze(0)
# encode only features with labels
encoded_images = autoencoder.encode(combined, labels)
# Plot the encoded real sample in a giga-plot
Plots.plot_rgb_gigaplot("ENCODED", encoded_images, n_classes, 5)

# evaluate with syntheval
# but first let's generate 700 fake samples
fake_samples, generated_fake_images = gan.sample(700, data.head(5), autoencoder, scaler)
fake_samples.to_csv('cgan_data.csv', index=False)


# generating synthetic data using synthcity
features["target"] = labels

# A conditional generative adversarial network which can handle tabular data.
syn_model = Plugins().get("ctgan")
syn_model.fit(features)
ctgan_data = syn_model.generate(count = 700)
ctgan_data.to_csv('ctgan_data.csv', index=False)

# A conditional VAE network which can handle tabular data.
syn_model = Plugins().get("tvae")
syn_model.fit(features)
tvae_data = syn_model.generate(count = 700)
tvae_data.to_csv('tvae_data.csv', index=False)