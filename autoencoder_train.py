import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: Convolutional layers to reduce the input dimensions
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # Output: (32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)  # Output: (64, 4, 4)
        )
        
        # Decoder: Upsample back to (28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1),  # Output: (32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,output_padding=1),  # Output: (1, 28, 28)
            nn.Sigmoid()  # Output values in the range [0, 1]
        )


    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed



def train(eval=True):
    # Prepare the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 15 
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, _ in train_loader:  # Labels are not needed for autoencoders
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), 'autoencoder_model.pth')


    if eval:
        import matplotlib.pyplot as plt
    
        model.eval()

        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device)
                outputs = model(images)
                break

        # Plot original and reconstructed images
        images = images.cpu().numpy()
        outputs = outputs.cpu().numpy()

        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            # Original images
            axes[0, i].imshow(images[i][0], cmap="gray")
            axes[0, i].axis("off")
            
            # Reconstructed images
            axes[1, i].imshow(outputs[i][0], cmap="gray")
            axes[1, i].axis("off")

        plt.show()


if __name__ == '__main__':
    train(eval=True)
