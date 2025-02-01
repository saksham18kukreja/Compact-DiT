"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from models import DiT_models
import argparse
from autoencoder_train import ConvAutoencoder
import time 

image_size = 28
model_variant = 'DiT-S/2'
model_function = DiT_models[model_variant]
num_classes = 10
# data_path = ""
batch_size = 128
epochs = 5
num_workers = 2
num_sampling_Steps = 500


def main():
    # Setup PyTorch:
    seed  = time.time()
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Load model:
    latent_size = image_size // 7
    model = model_function(input_size=latent_size, num_classes=num_classes).to(device)

    model.load_state_dict(torch.load("dit_model.pth",weights_only=True,map_location=device))
    model.eval()  # important!

    diffusion = create_diffusion(str(num_sampling_Steps))

    ae = ConvAutoencoder().to(device)
    ae.load_state_dict(torch.load("autoencoder_model.pth",weights_only=True,map_location=device))
    ae.eval() 

    # Labels to condition the model with (feel free to change):
    class_labels = [0,1,2,3,4,5,6,7,8,9]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 16, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)


    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([10] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=4.0)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = ae.decoder(samples).squeeze(1)
    
    # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        # Original images
        axes[i].imshow(samples[i], cmap="gray")
        axes[i].axis("off")
        
    plt.show()



if __name__ == "__main__":
    main()