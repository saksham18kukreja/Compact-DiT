import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
import argparse
import os
from tqdm import tqdm
from models import DiT_models
from diffusion import create_diffusion
from autoencoder_train import ConvAutoencoder

image_size = 28
model_variant = 'DiT-S/2'
model_function = DiT_models[model_variant]
num_classes = 10
# data_path = ""
batch_size = 128
epochs = 50
num_workers = 2

#function to train the model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #setup an experiment folder to save the checkpoints of the model

    #define the model here
    latent_size = image_size // 7
    model = model_function(input_size=latent_size, num_classes=num_classes).to(device)

    diffusion = create_diffusion(timestep_respacing="")

    ae = ConvAutoencoder().to(device)
    ae.load_state_dict(torch.load("autoencoder_model.pth",weights_only=True,map_location=device))

    #setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)

    #setup data loader
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])

    dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    model.train()

    #train the model
    for epoch in range(epochs):
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                data = ae.encoder(data)

            t = torch.randint(0, diffusion.num_timesteps, (data.shape[0],), device=device)
            model_kwargs = dict(y=label)
            loss_dict = diffusion.training_losses(model, data, t, model_kwargs)
            loss = loss_dict['loss'].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


            #log the loss values 

            #save the checkpoint of the model


        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(),"dit_model.pth")


if __name__ == "__main__":
    main()