import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# code from https://amaarora.github.io/2020/09/13/unet.html#u-net
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        self.batchnorm2d = nn.BatchNorm2d(in_ch)
    
    def forward(self, x):
        x = self.batchnorm2d(x)
        x = self.conv1(x)
        x = self.relu(x)
        return self.conv2(x)


class Encoder(nn.Module):
    def __init__(self, chs=(8,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 40)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)            
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(8,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], 40, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = F.interpolate(out, (256, 256))
        return out
    
    
class Block3(nn.Module):
    '''
    Module consisting of 3 convolutional layers
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2),  # first stride is always 2
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=3),  # constant kernel size from here
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        return self.layers(x)

    
class PointSetGen(nn.Module):
    def __init__(self, in_ch=8, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        
        # ====== ENCODER 1 ======     
        
        self.beginning = nn.Sequential(
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        
        self.block3_1 = Block3(16, 32)
        self.block3_2 = Block3(32, 64)
        self.block3_3 = Block3(64, 128)
        self.block3_4 = Block3(128, 256, kernel_size=5)
        self.upblock = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1))
        
        # ====== DECODER 1 ======
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.fully_connected1 = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(4, 2048),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(256)
        )
        
        self.skip1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            nn.Upsample((12, 12))
        )
        self.comb1 = nn.Conv2d(256, 256, kernel_size=3)
        self.blue1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5),
            nn.Upsample(scale_factor=2)
        )
        
        self.skip2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.Upsample((28, 28))
        )
        self.comb2 = nn.Conv2d(128, 128, kernel_size=3)
        self.blue2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5),
            nn.Upsample(scale_factor=2)
        )
        
        self.skip3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.Upsample((60, 60))
        )
        self.comb3 = nn.Conv2d(64, 64, kernel_size=3)
        self.blue3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.Upsample(scale_factor=2)
        )   
        
        self.skip4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.Upsample((124, 124))
        )
        self.comb4 = nn.Conv2d(32, 32, kernel_size=3)
        self.blue4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5),
            nn.Upsample(scale_factor=2)
        )  
        
        self.skip5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.Upsample((252, 252))
        )
        self.comb5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2)
        )
        
        # ====== ENCODER 2 ======
        
        self.enc_skip1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Upsample((124, 124))
        )
        self.enc_comb1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2)
        )
        
        self.enc_skip2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Upsample((60, 60))
        )
        self.enc_comb2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.enc_last2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        
        self.enc_skip3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Upsample((27, 27))
        )
        self.enc_comb3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.enc_last3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)  
        
        self.enc_skip4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.Upsample((11, 11))
        )
        self.enc_comb4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.enc_last4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        
        # ====== PREDICTOR ======

        self.fully_connected2 = nn.Linear(2048, 2048)
        self.fully_connected3 = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(9, 2048)
        )

        self.dec_blue1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)
        self.dec_skip1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.Upsample((9, 9))
        )
        self.convdeconv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.Upsample(scale_factor=2)
        )

        self.dec_skip2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Upsample((34, 34))
        )
        self.convdeconv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.Upsample(scale_factor=2)
        )
        
        self.dec_skip3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Upsample((134, 134))
        )
        self.convdeconv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.fully_connected4 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 300)
        )
        
        self.finalconv_full = nn.Conv2d(512, 64, kernel_size=1)
        self.finalconv_deconv = nn.Sequential(
            nn.Upsample((30,30))
        )
        
        self.mlp = nn.Conv2d(64, 40, kernel_size=1)

        
    def forward(self, x):
        
        # ====== ENCODER 1 ======
        x = self.beginning(x)
        x = self.block3_1(x)
        x1 = x    # can do this because torch returns new tensors for operations like nn.Conv2d
        
        # sequence of blocks of 3 convolutional layers
        x = self.block3_2(x) 
        x2 = x
        x = self.block3_3(x) 
        x3 = x
        x = self.block3_4(x) 
        x4 = x

        # substitute for block of 4 conv. layers b/c convolutions make images too small
        x = self.upblock(x)
        x5 = x
        
        # ====== DECODER 1 ======
        
        x = self.conv1(x)
        x_additional = self.fully_connected1(x)  # save for fully connected layer
        x = self.deconv1(x)
        
        x5 = self.skip1(x5)
        x = self.relu(torch.add(x, x5))  # torch.Size([1, 256, 12, 12])
        x = self.relu(self.comb1(x))
        x5 = x
        x = self.blue1(x)
        
        x4 = self.skip2(x4)
        x = self.relu(torch.add(x, x4))
        x = self.relu(self.comb2(x))
        x4 = x
        x = self.blue2(x)
        
        x3 = self.skip3(x3)
        x = self.relu(torch.add(x, x3))
        x = self.relu(self.comb3(x))
        x3 = x
        x = self.blue3(x)

        x2 = self.skip4(x2)
        x = self.relu(torch.add(x, x2))
        x = self.relu(self.comb4(x))
        x2 = x
        x = self.blue4(x)   
        
        x1 = self.skip5(x1)
        x = self.relu(torch.add(x, x1))
        x = self.comb5(x)
        
        # ====== ENCODER 2 ======
        # the function name and variable names should be off by 1
        x2 = self.enc_skip1(x2)
        x = self.relu(torch.add(x, x2))
        x = self.enc_comb1(x)
        
        x3 = self.enc_skip2(x3)
        x = self.relu(torch.add(x, x3))
        x = self.enc_comb2(x)
        x3 = x
        x = self.enc_last2(x)
        
        x4 = self.enc_skip3(x4)
        x = self.relu(torch.add(x, x4))
        x = self.enc_comb3(x)
        x4 = x
        x = self.enc_last3(x)
        
        x5 = self.enc_skip4(x5)
        x = self.relu(torch.add(x, x5))
        x = self.enc_comb4(x)
        x5 = x
        x = self.enc_last4(x)
        
        # ====== PREDICTOR ======
        
        x_additional = self.fully_connected2(x_additional)
        x_additional = self.relu(torch.add(x_additional, self.fully_connected3(x)))
        
        x = self.dec_blue1(x)
        x5 = self.dec_skip1(x5)
        x = self.relu(torch.add(x, x5))
        x = self.convdeconv1(x)
        
        x4 = self.dec_skip2(x4)
        x = self.relu(torch.add(x, x4))
        x = self.convdeconv2(x)
        
        x3 = self.dec_skip3(x3)
        x = self.relu(torch.add(x, x3))
        x = self.convdeconv3(x)
        
        x_additional = self.fully_connected4(x_additional) # torch.Size([1, 512, 600])
        x_additional = torch.reshape(x_additional, (self.batch_size, 512, 100, 3))
        x_additional = self.finalconv_full(x_additional)
        x = self.finalconv_deconv(x)
        x = torch.reshape(x, (self.batch_size, 64, 300, 3))
        x = torch.cat((x_additional, x), 2)
    
        uv = torch.meshgrid(torch.arange(0, 256), torch.arange(0, 256))
        uv = torch.stack(uv).permute(1,2,0).double().cuda() / torch.tensor(255)  # [256, 256, 2]
        xy = torch.sum(x.type(torch.float32), dim=1) # [self.batch_size, 40, 875, 3]

        img = torch.exp((torch.pow((uv[None,None,:,:,0]-xy[:,:,None,None,0]), 2) + torch.pow((uv[None,None,:,:,1]-xy[:,:,None,None,1]), 2)) / (torch.pow(xy[:,:,None,None,2], 2) + 1))  # [875,256,256]

        x = self.mlp(x)
        x = torch.sum(x, dim=-1)
        x = x[:,:,:,None,None] * img[:,None,:,:,:]
        x = torch.sum(x, dim=2)
        
        return x