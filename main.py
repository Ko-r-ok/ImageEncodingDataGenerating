import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import CGAN
import CAE
import Plots

# check if cuda is available and set the device accordingly
device = Plots.print_default()

BATCH_SIZE = 16
n_classes = 2

# reading the data
data = pd.read_csv('datasets/diabetes.csv')
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
autoencoder = CAE.Autoencoder(n_classes).to(device)
gan = CGAN.GAN(BATCH_SIZE, 28, 7, 7, 2).to(device)

# print the summary of the AE and G
autoencoder.get_summary()
print("")
gan.get_summary()

# pre-train the encoder
autoencoder.train_model(dataloader, epochs=100)

# gan training -- this also plots the training curves
gan.train_model(autoencoder, dataloader, epochs=1000)


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

# let's generate 700 fake samples
fake_samples, generated_fake_images = gan.sample(350, data.head(5), autoencoder, scaler)
fake_samples.to_csv('datasets/cgan_data.csv', index=False)