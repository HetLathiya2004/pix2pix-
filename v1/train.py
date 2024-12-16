import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        # Define the Laplacian kernel
        laplacian_kernel = torch.tensor([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
        
        # Replicate the kernel for RGB channels
        laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1).to(device)  # Shape: (3, 1, 3, 3)
        
        # Create a convolution layer with the Laplacian kernel
        self.laplacian_filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False, padding=1, groups=3)
        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False  # Freeze the kernel

    def forward(self, real_images, generated_images):
        """
        Compute the Laplacian-based L1 loss between real and generated RGB images.
        
        Args:
            real_images (torch.Tensor): The real images tensor of shape [batch_size, 3, height, width].
            generated_images (torch.Tensor): The generated images tensor of shape [batch_size, 3, height, width].

        Returns:
            torch.Tensor: The computed L1 loss.
        """
        # Apply the Laplacian filter to the real and generated images
        real_filtered = self.laplacian_filter(real_images)
        generated_filtered = self.laplacian_filter(generated_images)
        
        # Compute L1 loss between filtered images
        loss = F.l1_loss(generated_filtered, real_filtered)
        return loss



def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce,lp, g_scaler, d_scaler,epoch
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        print(x.type())
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            lp_loss = 50*lp(y,y_fake)
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1 + lp_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        LP_loss = lp_loss.mean()
        L1_loss = L1.mean()
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                LP_loss = LP_loss.item(),
                L1_loss = L1_loss.item(),
                epoch = epoch
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    lp = LaplacianLoss()
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        save_some_examples(gen, val_loader, epoch, folder="evaluation")
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,lp, g_scaler, d_scaler,epoch
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        


if __name__ == "__main__":
    main()