import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Load the dataset and preprocess images using PyTorch Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # If your images are RGB make sure to convert them to grayscale
    transforms.Resize((54, 54)),  # Resize to the desired size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] if your images are [0, 255]
])

# Use an absolute path or the correct relative path
dataset_root = "Laptop_Locked_Samples/"
assert os.path.exists(dataset_root), f"Couldn't find the directory: {dataset_root}"

dataset = ImageFolder(root=dataset_root, transform=transform)

dataloader = DataLoader(dataset, batch_size = 9, shuffle=True)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is a noise vector
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            # Output is a generated image
            nn.Linear(1024, 8748), # Adjusted for 54x54 RGB images
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is an image
            nn.Linear(8748, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output is a probability that the input is real
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the networks
generator = Generator()
discriminator = Discriminator()

# Set up the loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
num_epochs = 10000
batch_size = 9  # batch_size is set globally here, it's not necessary to set it inside the loop

# This will iterate through the dataset num_epochs times.
for epoch in range(num_epochs):
    # 1. Train the Discriminator
    for i, (real_data, _) in enumerate(dataloader):
        discriminator.zero_grad()
        
        # 1.1 Train Discriminator on real data
        current_batch_size = real_data.size(0)  # dynamically get the batch size
        label = torch.full((current_batch_size, 1), 1.0, dtype=torch.float)  # using real_data.size(0) to handle dynamic batch sizes
        print(real_data.shape)  # should print torch.Size([batch_size, 3, 54, 54])
        real_data = real_data.view(current_batch_size, -1)  # Flatten the real_data
        output = discriminator(real_data)
        errD_real = criterion(output, label)
        errD_real.backward()

        # 1.2 Train Discriminator on fake data
        noise = torch.randn(current_batch_size, 100)  # Adjust noise to current_batch_size
        fake = generator(noise)
        label.fill_(0)  # Set label to 0 for fake data
        output = discriminator(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()

        # Update Discriminator
        errD = errD_real + errD_fake
        optimizerD.step()
        
        # 2. Train the Generator
        generator.zero_grad()
        label.fill_(1)  # Generator wants the discriminator to output 1 for fake data
        output = discriminator(fake)  # Don't detach here
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

    # Print statistics for every epoch
    print(f"[{epoch}/{num_epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}")
