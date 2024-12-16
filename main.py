import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms
import glob
from PIL import Image
import os
import pytorch_lightning as pl
from torch import nn, sigmoid

batchsize = 64
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable
import torch.autograd as autograd

imgs_path = glob.glob('data/train/*.png') 

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((256, 512)),  
    transforms.Normalize(0.5, 0.5) ])

class Anime_dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.imgs_path = imgs_path

    def __getitem__(self, index): 
        img_path = self.imgs_path[index]
        pil_img = Image.open(img_path) 
        pil_img = pil_img.convert('RGB') 
        pil_img = transform(pil_img) 
        w = pil_img.size(2) // 2 
        return pil_img[:, :, w:], pil_img[:, :, :w] 

    def __len__(self):
        return len(self.imgs_path)


dataset = Anime_dataset(imgs_path)

dataloader = data.DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True,num_workers=10)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True,negative_slope=0.2), 
        )
        
    def forward(self, x, is_bn=True):
        x = self.conv_relu(x) 
        # if is_bn: 
        #     x = self.bn(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu = nn.Sequential(  
            nn.ConvTranspose2d(in_channels, out_channels, 
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True,negative_slope=0.2),  
        )

    def forward(self, x, is_drop=False):
        x = self.upconv_relu(x) 
        # x = self.bn(x) 
        # if is_drop:
        #     x = F.dropout2d(x,p = 0.1)
        return x


class Generator(nn.Module): 
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Downsample(3, 64) 
        self.down2 = Downsample(64, 128)  
        self.down3 = Downsample(128, 256)  
        self.down4 = Downsample(256, 512)  
        self.down5 = Downsample(512, 512)  
        self.down6 = Downsample(512, 512) 

        self.up1 = Upsample(512, 512)  
        self.up2 = Upsample(1024, 512)  
        self.up3 = Upsample(1024, 256) 
        self.up4 = Upsample(512, 128)  
        self.up5 = Upsample(256, 64) 

        self.last = nn.ConvTranspose2d(128, 3,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1)

    def forward(self, x): 
        x1 = self.down1(x) 
        x2 = self.down2(x1)  
        x3 = self.down3(x2) 
        x4 = self.down4(x3) 
        x5 = self.down5(x4) 
        x6 = self.down6(x5)  
        x6 = self.up1(x6, is_drop=True)  
        x6 = torch.cat([x6, x5], dim=1)  
        x6 = self.up2(x6, is_drop=True)  
        x6 = torch.cat([x6, x4], dim=1)  
        x6 = self.up3(x6, is_drop=True)  
        x6 = torch.cat([x6, x3], dim=1)  
        x6 = self.up4(x6)  
        x6 = torch.cat([x6, x2], dim=1)  

        x6 = self.up5(x6)  
        x6 = torch.cat([x6, x1], dim=1)  
        x = torch.tanh(self.last(x6))  
        return x


class Discriminator(nn.Module): 
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.down1 = Downsample(6, 64)  
        self.down2 = Downsample(64, 128)  
        self.cov1 = nn.Conv2d(128, 256, kernel_size=3) 
        self.bn = nn.InstanceNorm2d(256, affine=True)
        self.bn1 = nn.InstanceNorm2d(128, affine=True)
        self.last = nn.Conv2d(256, 1, kernel_size=3)  
        self.flat = nn.Flatten()
        self.lnear_1 = nn.Linear(128,1)
        # self.lnear_2 = nn.Linear(60,1)
    def forward(self, img, mask): 
        x = torch.cat([img, mask], dim=1) 
        x = self.down1(x, is_bn=False) 
        x = self.down2(x)
        x = F.dropout2d(F.leaky_relu(self.bn(self.cov1(x)),negative_slope=0.2))
        x = F.dropout2d(F.leaky_relu(self.last(x),negative_slope=0.2))
        #print(f"x shape {x.shape}")
        # m = nn.AdaptiveAvgPool2d(1)
        # x = m(x)
        # #print(f"x shape {x.shape}")
        # x = self.flat(x)
        # #print(f"x shape {x.shape}")
        # x = self.lnear_1(x)
        x = torch.sigmoid(x)
        #print(f"x shape {x.shape}")
        # x = self.lnear_1(x)
        # x = self.lnear_2(x) 
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = Generator().to(device)  
D = Discriminator().to(device)  

# state_dict = torch.load('generator_epoch_56.pth',weights_only=True)
# G.load_state_dict(state_dict)
# dis_stat = torch.load('discriminator_epoch_56.pth',weights_only=True)
# D.load_state_dict(dis_stat)


loss1 = torch.nn.BCELoss()
loss2 = torch.nn.L1Loss()
LAMBDA = 7  

imgs_path_test = glob.glob('data/val/*.png') 
test_dataset = Anime_dataset(imgs_path_test)
test_dataloader = data.DataLoader(test_dataset, batch_size=batchsize,num_workers=50)
imgs_batch, masks_batch = next(iter(test_dataloader)) 
imgs_batch = imgs_batch.to(device)
masks_batch = masks_batch.to(device)

def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)


def compute_pj(x, mu_k, K, L, W):
    # print(f"x is on device {x.device},{mu_k.device} ")
    # We assume that x has only one channel already.
    # Flatten spatial dims.
    x.to(device)
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1).to()  # Construct K channels.

    # Apply activation functions.
    return phi_k(x - mu_k, L, W)


class HistLayerBase(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = 255
        self.L = 1 / self.K  # 2 / K if values are in [-1,1] (as per the paper).
        self.W = self.L / 2.5

        # Define mu_k and move it to the correct device.
        self.mu_k = (self.L * (torch.arange(self.K, device=self.device) + 0.5)).view(-1, 1)


class SingleDimHistLayer(HistLayerBase):
    def __init__(self, device=None):
        super().__init__(device=device)

    def forward(self, x):
        # Ensure the input tensor is on the correct device.
        x = x.to(self.device)
        N = x.size(1) * x.size(2)
        pj = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        return pj.sum(dim=2) / N


class JointHistLayer(HistLayerBase):
    def __init__(self, device=None):
        super().__init__(device=device)

    def forward(self, x, y):
        # Ensure both input tensors are on the correct device.
        x = x.to(self.device)
        y = y.to(self.device)
        N = x.size(1) * x.size(2)
        p1 = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        p2 = compute_pj(y, self.mu_k, self.K, self.L, self.W)
        return torch.matmul(p1, torch.transpose(p2, 1, 2)) / N


class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x, y):
        # Automatically infer device from inputs
        device = self.device

        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins, device=device)
        s, t = torch.meshgrid(r, r, indexing='ij')
        tt = t >= s

        cdf_x = torch.matmul(x, tt.float())
        cdf_y = torch.matmul(y, tt.float())

        return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, p1, p2, p12):
        # Automatically infer device from inputs
        device = self.device

        # input p12 has dims: (Batch x Bins x Bins)
        # input p1 & p2 has dims: (Batch x Bins)

        product_p = torch.matmul(
            torch.transpose(p1.unsqueeze(1), 1, 2), 
            p2.unsqueeze(1)
        ) + torch.finfo(p1.dtype).eps
        mi = torch.sum(p12 * torch.log(p12 / product_p + torch.finfo(p1.dtype).eps), dim=(1, 2))
        h = -torch.sum(p12 * torch.log(p12 + torch.finfo(p1.dtype).eps), dim=(1, 2))

        return 1 - (mi / h)

class ColourLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.earth = EarthMoversDistanceLoss().to(self.device)
        self.hist = SingleDimHistLayer().to(self.device)
    def forward(self,img,img_t):
        
        r,g,b = img[:,0],img[:,1],img[:,2]
        r_t,g_t,b_t = img_t[:,0],img_t[:,1],img_t[:,2]
        r_hist = self.hist(r)
        g_hist = self.hist(g)
        b_hist = self.hist(b)

        r_t_hist = self.hist(r_t)
        g_t_hist = self.hist(g_t)
        b_t_hist = self.hist(b_t)
        #+
        loss = self.earth(r_hist,r_hist) +  self.earth(g_hist,g_t_hist) + self.earth(b_hist,b_t_hist)

        return loss


def gen_img_plot(model, img, mask, save_dir="output_images"):
    import os
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    y = model(img).permute(0, 2, 3, 1).detach().cpu().numpy()
    print(y.shape)
    predictions = y   
    img = img.permute(0, 2, 3, 1).cpu().numpy()
    mask = mask.permute(0, 2, 3, 1).cpu().numpy() 
    
    for j in range(16):
        plt.figure(figsize=(10, 10))
        display_list = [img[j], mask[j], predictions[j]]
        title = ['Input', 'Truth', 'Output']
        
        for i in range(3):  
            plt.subplot(1, 3, i + 1) 
            plt.title(title[i])
            plt.imshow((display_list[i] + 1) / 2)
            plt.axis('off')  
        
        # Save the figure as an image
        output_path = os.path.join(save_dir, f"plot_{j}.png")
        plt.savefig(output_path)
        plt.close()  # Close the figure to free memory
        # print(f"Saved plot {j} to {output_path}")


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

lp = LaplacianLoss()
class GANModule(pl.LightningModule):
    def __init__(self, generator, discriminator, dataloader, lambda_gp=10, lambda_l1=10, save_dir="checkpoints"):
        super().__init__()
        self.G = generator
        self.D = discriminator
        self.dataloader = dataloader
        self.lambda_gp = lambda_gp
        self.lambda_l1 = lambda_l1
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Create save directory if it doesn't exist
        self.save_hyperparameters()
        self.num_iter = 0

        # Disable automatic optimization
        self.automatic_optimization = False

    def forward(self, imgs):
        return self.G(imgs)

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        BATCH_SIZE, C, H, W = real_samples.shape
        device = real_samples.device
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(real_samples,interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        imgs, masks = imgs.to(self.device), masks.to(self.device)
        # Get optimizers
        opt_D, opt_G = self.optimizers()
        D_loss = 0
        D_y_loss = 0
  # Update Discriminator weights
        # --- Train Generator ---
        # --- Train Discriminator ---
        D_loss = 0
        D_fake_loss = 0
        D_real_loss = 0
        opt_D.zero_grad()  # Zero gradients for Discriminator
        D_real = self.D(imgs, masks)
        G_img = self.G(imgs)
        D_fake = self.D(imgs, G_img.detach())
            
        D_real_loss = loss1(D_real, torch.ones_like(D_real))
        self.manual_backward(D_real_loss, retain_graph=True) 
            
        D_fake_loss = loss1(D_fake, torch.zeros_like(D_fake))
        D_loss = D_real_loss + D_fake_loss
        self.manual_backward(D_loss)  # Backpropagation for Discriminator
        opt_D.step()

        for _ in range(1):
            opt_G.zero_grad()  # Zero gradients for Generator
            G_img = self.G(imgs)
            D_fake = self.D(imgs, G_img)
            G_loss_BEC = loss1(D_fake, torch.ones_like(D_fake))
            G_loss_L1 = loss2(G_img, masks)  # Assuming `loss2` is defined elsewhere
            G_loss_lp = lp(masks,G_img)
            G_loss =  G_loss_BEC + 3*G_loss_lp + 7*G_loss_L1
        
            self.manual_backward(G_loss)  # Backpropagation for Generator
            opt_G.step()  
                 
            self.log_dict({
                            "D_l": D_loss,
                            "Dfl":D_fake_loss,
                            "Drl":D_real_loss,
                            "G_BEC": G_loss_BEC,
                            "G_L1": G_loss_L1,
                            "G_lp":G_loss_lp,
                            "G_l": G_loss,
                            "idx":batch_idx
                        },prog_bar=True)

        # self.num_iter = self.num_iter + 1
        # Logging losses

    def on_train_epoch_end(self):
        """Save the generator and discriminator weights after every epoch."""
        gen_path = os.path.join(self.save_dir, f"generator_epoch_{self.current_epoch}.pth")
        disc_path = os.path.join(self.save_dir, f"discriminator_epoch_{self.current_epoch}.pth")
        gen_img_plot(G, imgs_batch, masks_batch)
        torch.save(self.G.state_dict(), gen_path)
        torch.save(self.D.state_dict(), disc_path)
        print(f"Saved generator to {gen_path} and discriminator to {disc_path}")

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        return [opt_D, opt_G]

    def train_dataloader(self):
        return self.dataloader



if __name__ == "__main__":
    
    model = GANModule(G,D, dataloader, save_dir="model_checkpoints").to(device)
    print(device)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator ="gpu",
        devices=1,
        # Set the number of GPUs or None for CPU
    )

    trainer.fit(model)


