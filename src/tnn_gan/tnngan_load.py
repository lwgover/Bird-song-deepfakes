import torch
# Initialize the generator and discriminator (same as before)
noise_dim = 100
condition_dim = 10
output_channels = 2
img_size = (128, 128)

generator = Generator(noise_dim, condition_dim, output_channels, img_size)
discriminator = Discriminator(condition_dim, output_channels, img_size)

# Load the saved weights
generator.load_state_dict(torch.load(generator_weights_path))
discriminator.load_state_dict(torch.load(discriminator_weights_path))

# Set the models to evaluation mode
generator.eval()
discriminator.eval()

# Optional: move the models to GPU if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

def generate_images(generator, num_images, noise_dim, condition_dim, device):
    # Generate noise and condition vectors
    noise = torch.randn(num_images, noise_dim).to(device)
    conditions = torch.randn(num_images, condition_dim).to(device)

    # Generate images using the generator
    with torch.no_grad():
        generated_images = generator(noise, conditions)

    return generated_images

num_images_to_generate = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generated_images = generate_images(generator, num_images_to_generate, noise_dim, condition_dim, device)

def show_images(images, nrow=2, ncol=5):
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2  # Scale pixel values back to [0, 1]
        ax.imshow(img, cmap="gray" if img.shape[2]==1 else None)
        ax.axis("off")
    plt.show()

generated_images_np = generated_images.cpu().numpy()
show_images(generated_images_np)