import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
import imageio
import os

from dataloader import FilteredDataset
from networks import Generator, Discriminator


def test_discriminator():
    disc = Discriminator()
    disc.summary()

    test_input = torch.randn(32, 3, 128, 128)
    test_output = disc(test_input).detach()
    print(test_output)


def test_generator(generator=None, latent_dim=128, device='cuda'):
    if generator is None:
        generator = Generator()
        generator.summary()

    test_input = torch.randn(32, latent_dim).to(device)
    test_output = generator(test_input).detach()
    save_image(test_output, "generator_output.png")


def train(generator, discriminator, dataloader, batch_size,
          latent_dim=128, log_iter=200, device='cuda'):
    loss_fn = nn.BCELoss()
    real_label = torch.ones((batch_size, 1), device=device)
    fake_label = torch.zeros((batch_size, 1), device=device)

    Path("./output").mkdir(exist_ok=True)
    progress = tqdm(total=len(dataloader))
    sum_gen_loss = 0

    for i, images in enumerate(dataloader):
        discriminator.zero_grad()
        images = images.to(device)

        # backward real images
        output = discriminator(images)
        loss_real = loss_fn(output, real_label)
        loss_real.backward()

        # backward fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake = generator(noise)
        output = discriminator(fake.detach())
        loss_fake = loss_fn(output, fake_label)
        loss_fake.backward()

        # apply on discriminator
        discriminator.optim.step()

        # train generator
        generator.zero_grad()
        output = discriminator(fake)
        loss_gen = loss_fn(output, real_label)
        loss_gen.backward()
        generator.optim.step()

        # log
        sum_gen_loss += loss_gen.item()
        progress.update(1)
        progress.set_description(str(sum_gen_loss))

        if i % log_iter == 0:
            save_image(fake, f"./output/{i}.png")

    images = []
    for filename in os.listdir("./output"):
        images.append(imageio.imread(f"./output/{filename}"))
    imageio.mimsave("output.gif", images)


if __name__ == "__main__":
    batch_size = 32
    latent_dim = 128
    img_size = 128

    torch.manual_seed(42)

    dataset = FilteredDataset(img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()

    generator.summary()
    discriminator.summary()

    train(generator, discriminator, loader, batch_size=batch_size)
    test_generator(generator, latent_dim=latent_dim)
