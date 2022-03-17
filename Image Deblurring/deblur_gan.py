import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision import transforms

from torchsummary import summary

from vgg_perceptual_loss import VGGPerceptualLoss

import time
import cv2
import os

'''
Deep convolutional generative adversarial network for image deblurring.  The generator architecture 
is based on the residual learning architecture described here: https://arxiv.org/pdf/1512.03385.pdf.
Originally trained on the GOPRO dataset.
'''

batch_size = 16
img_size = (3, 720, 1280)
lr = 0.0001
b1, b2 = 0.7, 0.999  # Beta parameters for Adam optimizer
n_epochs = 100

# This apparently makes pytorch faster
# See https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# First, we load the data.
# This assumes the images are split into four directories called train_blur, train_sharp,
# valid_blur, and valid_sharp, and that the images are named 'i.png' with i being the 
# image number, ranging from 0 to one less than the number of images. Image 'n' in the 
# blurred image directory corresponds to image 'n' in the sharp image directory.
def listdir_nohidden(path: str):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

class GoproData:
    def __init__(self):
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return self.max_im + 1
    
    def get_item_at_index(self, index: int, dataset: str):
        blur_img = cv2.imread(f'{dataset}_blur/{index}.png')
        #blur_gamma_img = cv2.imread(f'train_blur_gamma/{index}.png')
        sharp_img = cv2.imread(f'{dataset}_sharp/{index}.png')
        return self.transform(blur_img), self.transform(sharp_img)

class TrainData(Dataset, GoproData):
    def __init__(self):
        super().__init__()
        self.max_im = int(max(listdir_nohidden('./train_blur/'), key=lambda s: int(s[:-4]))[:-4])

    def __getitem__(self, index):
        return self.get_item_at_index(index, 'train')

# class ValidData(Dataset, GoproData):
#     def __init__(self):
#         super().__init__()
#         self.max_im = int(max(listdir_nohidden('./valid_blur/'), key=lambda s: int(s[:-4]))[:-4])

#     def __getitem__(self, index):
#         return self.get_item_at_index(index, 'valid')

class ResBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            filters, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            padding_mode='reflect',
            use_dropout=False
        ):
        super().__init__()

        block = [
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=filters, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                padding_mode=padding_mode,
                bias=False
            ),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=filters, 
                out_channels=filters, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                padding_mode=padding_mode,
                bias=False
            ),
            nn.BatchNorm2d(filters)
        ]
        if use_dropout:
            block.insert(3, nn.Dropout(p=0.5))
        self.res_block = nn.Sequential(*block)
    
    def forward(self, input):
        return self.res_block(input) + input

class Generator(nn.Module):
    def __init__(self, input_shape, ngf, n_res_blocks, n_downsampling=2, use_dropout=False):
        super().__init__()

        # Downsample the image
        self.downsampling = [
            nn.Conv2d(
                in_channels=input_shape[1], 
                out_channels=ngf, 
                kernel_size=7, 
                stride=1,
                padding=3, 
                padding_mode='reflect',
                bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        ]
        for i in range(n_downsampling):
            self.downsampling.extend([
                nn.Conv2d(
                    in_channels=ngf*2**i, 
                    out_channels=ngf*2*2**i, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    padding_mode='reflect',
                    bias=False
                ),
                nn.BatchNorm2d(ngf*2*2**i),
                nn.ReLU()
            ])
        self.downsampling = nn.Sequential(*self.downsampling)

        # Apply n_res_blocks number of ResNet blocks
        mult = 2**n_downsampling
        self.res_blocks = nn.Sequential(*[ResBlock(
            in_channels = ngf*mult, 
            filters = ngf*mult, 
            use_dropout = use_dropout
        ) for _ in range(n_res_blocks)])

        # Upsample the image to revert its shape
        self.upsampling = []
        for i in range(n_downsampling):
            self.upsampling.extend([
                nn.ConvTranspose2d(
                    in_channels=ngf*2**(n_downsampling - i),
                    out_channels= ngf*2**(n_downsampling - i - 1),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(ngf*2**(n_downsampling - i - 1)),
                nn.ReLU()
            ])
        self.upsampling.extend([
            nn.Conv2d(
                in_channels=ngf,
                out_channels=input_shape[1],
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode='reflect'
            ),
            nn.Sigmoid()
        ])
        self.upsampling = nn.Sequential(*self.upsampling)

    def forward(self, input):
        x = self.downsampling(input)
        x = self.res_blocks(x)
        x = self.upsampling(x)
        x = torch.add(x, input)
        x = torch.div(x, 2)
        return x     

class Discriminator(nn.Module):
    def __init__(self, ndf, n_layers=3, use_sigmoid=True, negative_slope=0.2):
        super().__init__()

        self.conv_layers = [
            nn.Conv2d(
                in_channels=3,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(negative_slope)
        ]
        for i in range(n_layers):
            mult = min(2**(i + 1), 8)
            self.conv_layers.extend([
                nn.Conv2d(
                    in_channels=ndf*mult//2,
                    out_channels=ndf*mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(ndf*mult),
                nn.LeakyReLU(negative_slope)
            ])
        mult = min(2**n_layers, 8)
        self.conv_layers.extend([
            nn.Conv2d(
                in_channels=ndf*mult,
                out_channels=ndf*mult,
                kernel_size=4,
                stride=1,
                padding='same',
                bias=False
            ),
            nn.BatchNorm2d(ndf*mult),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(
                in_channels=ndf*mult,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding='same'
            )
        ])
        if use_sigmoid:
            self.conv_layers.append(nn.Sigmoid())
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.lin_layers = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.flatten(1)
        x = self.lin_layers(x)
        return x

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model.eval()

def main(verbose=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('Using CUDA.' if torch.cuda.is_available() else 'CUDA not available.')

    if verbose: print('Loading data...')
    train_loader = DataLoader(
        TrainData(), 
        batch_size=batch_size, 
        shuffle=False,
        # num_workers=1,
        pin_memory=True
    )
    # valid_loader = DataLoader(
    #     ValidData(), 
    #     batch_size=batch_size, 
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True
    # )

    # Loss function
    adversarial_loss = nn.BCELoss()

    if verbose: print('Defining models...')
    G = Generator(
        input_shape=[batch_size, *img_size],
        ngf=64,
        n_res_blocks=9,
        n_downsampling=2,
        use_dropout=True
    )
    D = Discriminator(
        ndf=64,
        n_layers=3,
        use_sigmoid=True
    )
    G.to(device)
    D.to(device)

    if verbose:
        print('Generator:')
        summary(G, img_size)
        print('\nDiscriminator:')
        summary(D, img_size)

    # Optimizers
    optimizer_G = Adam(G.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = Adam(D.parameters(), lr=lr, betas=(b1, b2))

    vgg_loss = VGGPerceptualLoss()
    try:
        G.train()
        for epoch in range(n_epochs):
            if verbose: print(f'Beginning epoch {epoch}/{n_epochs}')
            for i, imgs in enumerate(train_loader):
                blur, sharp = imgs
                blur, sharp = blur.to(device), sharp.to(device)
                
                # Adversarial ground truths
                real = lambda t: torch.ones_like(t)
                fake = lambda t: torch.zeros_like(t)
                
                # ----------------
                # Train Generator
                # ----------------
                if verbose: print(f'Training generator on batch {i}/{len(train_loader)}...')
                t = time.time()
                
                # optimizer_G.zero_grad()
                for param in G.parameters():
                    param.grad = None
                
                # Generate a batch of images
                gen_imgs = G(blur)
                
                # Loss measures generator's ability to fool the discriminator
                D.eval()
                disc_pred_fake = D(gen_imgs)
                G_adversarial_loss = adversarial_loss(disc_pred_fake, real(disc_pred_fake))
                G_vgg_loss = vgg_loss(gen_imgs, sharp)
                G_loss = G_adversarial_loss + G_vgg_loss

                G_loss.backward()
                optimizer_G.step()

                if verbose: print(f'  Trained generator on batch {i} in {(time.time() - t):.2f} seconds')
                
                # --------------------
                # Train Discriminator
                # --------------------
                if verbose: print(f'Training discriminator on batch {i}/{len(train_loader)}...')
                t = time.time()

                D.train()
                # optimizer_D.zero_grad()
                for param in D.parameters():
                    param.grad = None
                
                # Measure discriminator's ability to classify real from generated samples
                disc_pred_real = D(sharp)
                real_loss = adversarial_loss(disc_pred_real, real(disc_pred_real))
                disc_pred_fake = D(gen_imgs.detach())
                fake_loss = adversarial_loss(disc_pred_fake, fake(disc_pred_fake))
                D_loss = (real_loss + fake_loss) / 2
                
                D_loss.backward()
                optimizer_D.step()

                if verbose: print(f'  Trained discriminator on batch {i} in {(time.time() - t):.2f} seconds')
                
                # ----------------
                # Log Progress
                # ----------------
                
                print(f'[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}]\n  [D loss: {D_loss.item():.4f}] [G loss (adv): {G_adversarial_loss.item():.4f}] [G loss (vgg): {G_vgg_loss.item():.4f}]')
        
        save_model(G, './deblur_gan_generator.pt')

    except KeyboardInterrupt:
        # Do this so we still save the model if we stop training early
        save_model(G, './deblur_gan_generator.pt')

if __name__ == '__main__':
    main()