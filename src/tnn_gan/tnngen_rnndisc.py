import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer,TransformerEncoder

class Generator(nn.Module):
    def __init__(self, noise_dim:int, condition_dim:int, output_channels=2, img_size=(128, 128)):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_channels = output_channels
        self.img_size = img_size

        self.fc = nn.Linear(noise_dim + condition_dim, img_size[0] * img_size[1] // 16)

        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, img_size[0] // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Load GPT weights and initialize a transformer layer
        self.transformer_layer = TransformerEncoderLayer(d_model=img_size[0] // 4, nhead=4)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=2)


        # Upsampling layers with transposed 2D convolutional layers and batch normalization
        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(img_size[0] // 4, img_size[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(img_size[0] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(img_size[0] // 2, img_size[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(img_size[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(img_size[0], output_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # Final convolutional layer and tanh activation
        self.output_layer = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        x = self.fc(x)
        x = x.view(-1, self.img_size[0] // 4, self.img_size[1] // 4, 1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        
        # Reshape and apply the transformer layer
        x = x.permute(0, 2, 3, 1) ## need to shape (batch_size, seqlength, featuredim)
        x = self.transformer(x)
        x = x.permute(0, 3, 1, 2) ## reshape back

        x = self.up_layers(x)
        x = self.output_layer(x)

        return x
    
    
class Discriminator(nn.Module):

    def __init__(self, condition_dim, output_channels=2, img_size=(128, 128)):
        super(Discriminator, self).__init__()

        self.condition_dim = condition_dim
        self.output_channels = output_channels
        self.img_size = img_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,32, kernel_size = (5, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), padding='same'),
            nn.Conv2d(32,64, kernel_size = (5, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), padding='same'),
            nn.Conv2d(64,128, kernel_size = (5, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), padding='same'),
            nn.Conv2d(128,256, kernel_size = (5, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), padding='same'),
            nn.Conv2d(256,256,kernel_size = (5, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), padding='same'),
            nn.Conv2d(256,256, kernel_size = (5, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 1), padding='same'),
        )
        # GRU layer
        self.gru_layer = nn.GRU(input_size=img_size[0]*2, hidden_size=img_size[0]*2, num_layers=2)
        self.gru = nn.GRU(input_size=img_size[0]*2, hidden_size=img_size[0]*2, num_layers=2, batch_first=True)

        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=img_size[0]*2, out_features=1)
        )
    def forward(self, x, condition):
        x = self.conv_layers(x).view(-1, 256)

        # Reshape and apply the transformer layer
        x = x.permute(0, 2, 3, 1)
        x = self.gru(x)
        x = x.permute(0, 3, 1, 2)

        x = x.view(x.size(0), -1)
        x = torch.cat((x, condition), dim=1)
        x = torch.max(x, 1)[0]
        x = self.fc(x)

        return x

if __name__ == "__main__":
    noise_dim = 100
    condition_dim = 10
    output_channels = 2
    img_size = (128, 128)

    generator = Generator(noise_dim, condition_dim, output_channels, img_size)
    discriminator = Discriminator(condition_dim, output_channels, img_size)

    # Hyperparameters
    lr = 0.0002
    beta1 = 0.5
    num_epochs = 200
    batch_size = 64

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # GPU support
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        criterion.cuda()

    for epoch in range(num_epochs):
        for i, (real_images, conditions) in enumerate(dataloader):
            batch_size = real_images.size(0)

            # Labels for real and fake images
            real_label = Variable(torch.ones(batch_size, 1).cuda())
            fake_label = Variable(torch.zeros(batch_size, 1).cuda())

            # Train the discriminator
            optimizer_D.zero_grad()

            real_images = Variable(real_images.cuda())
            conditions = Variable(conditions.cuda())

            real_output = discriminator(real_images, conditions)
            real_loss = criterion(real_output, real_label)

            noise = Variable(torch.randn(batch_size, noise_dim).cuda())
            fake_images = generator(noise, conditions)

            fake_output = discriminator(fake_images.detach(), conditions)
            fake_loss = criterion(fake_output, fake_label)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train the generator
            optimizer_G.zero_grad()

            fake_output = discriminator(fake_images, conditions)
            g_loss = criterion(fake_output, real_label)
            g_loss.backward()
            optimizer_G.step()

        # Print losses
        print(f'Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}')

    generator_weights_path = 'generator_weights.pth'
    discriminator_weights_path = 'discriminator_weights.pth'

    torch.save(generator.state_dict(), generator_weights_path)
    torch.save(discriminator.state_dict(), discriminator_weights_path)
