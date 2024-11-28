# ejemplo de https://github.com/pytorch/examples

from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class DiscriminatorDetalle(nn.Module):
    def __init__(self, ndf, nc):
        super(DiscriminatorDetalle, self).__init__()
        
        self.cv0 = nn.Conv2d(nc, ndf, 4, (2,1), (1,0), bias=False)

        self.cv1 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)
        self.bt1 = nn.BatchNorm2d(ndf * 2)

        self.cv2 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)
        self.bt2 = nn.BatchNorm2d(ndf * 4)

        self.cv3 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False)
        self.bt3 = nn.BatchNorm2d(ndf * 8)

        self.cv4 = nn.Conv2d(ndf * 8, 1, (2,1), 1, 0, bias=False)

        self.lek = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()
        

    def forward(self, input):        
        x = self.cv0(input)
        x = self.lek(x)
        
        x = self.cv1(x)
        x = self.lek(self.bt1(x))
        
        x = self.cv2(x)
        x = self.lek(self.bt2(x))
        
        x = self.cv3(x)
        x = self.lek(self.bt3(x))
        
        x = self.cv4(x)
        return self.sig(x).view(x.size(0))  


class GeneratorDetalle(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(GeneratorDetalle, self).__init__()
        
        self.ct0 = nn.ConvTranspose2d(     nz, ngf * 8, 2, 1, 0, bias=False)
        self.bt0 = nn.BatchNorm2d(ngf * 8)

        self.ct1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bt1 = nn.BatchNorm2d(ngf * 4)

        self.ct2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bt2 = nn.BatchNorm2d(ngf * 2)

        self.ct3 = nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False)
        self.bt3 = nn.BatchNorm2d(ngf)

        self.ct4 = nn.ConvTranspose2d(    ngf,      nc, (4,2), (2,1), 1, bias=False)
        
        self.elu = nn.ELU(alpha=0.1, inplace=True) 
        self.tan = nn.Tanh()

    def forward(self, input):
        
        x = self.ct0(input)
        x = self.elu(self.bt0(x))
        
        x = self.ct1(x)
        x = self.elu(self.bt1(x))

        x = self.ct2(x)
        x = self.elu(self.bt2(x))

        x = self.ct3(x)
        x = self.elu(self.bt3(x))

        x = self.ct4(x)

        return self.tan(x)
    

class LPRdataset(Dataset):
    def __init__(self, root_path, transform):
        self.root_path = root_path
        self.transform = transform
        self.images = []
        
        for k in range(10):
            img_digits = sorted([os.path.join(root_path, '%d' % k, i) for i in os.listdir(root_path + "/%d/" % k)])
            self.images.extend(img_digits)
            

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        folders = os.path.dirname(self.images[index])
        label = folders.split(os.path.sep)[-1:][0]
        
        return self.transform(img), int(label)

    def __len__(self):
        
        return len(self.images)
 

outf = 'out'
if os.path.exists(outf) == False:
    os.mkdir(outf)

manualSeed = 1000
imageSize = (32,15)
batchSize = 16

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

dataroot = 'BaseOCR_MultiStyle'
if os.path.exists(dataroot) == False:
    os.mkdir(dataroot)

dataset = LPRdataset(dataroot, transform=transforms.Compose([
                           transforms.Resize(imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

assert dataset
dataloader = DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=0)

nc=3
device = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
nz      = 100
ngf     = 32
ndf     = 32

############################################################################
netG = GeneratorDetalle(nz, ngf, nc).to(device)
netG.apply(weights_init)
print(netG)

netD = DiscriminatorDetalle(ndf, nc).to(device)
netD.apply(weights_init)
print(netD)
############################################################################

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

netG.train()
netD.train()

niter = 30

for epoch in range(niter):
    with torch.no_grad():
        fake = netG(fixed_noise)
    vutils.save_image(fake.detach(),
            '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
            normalize=True)

    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        ########################
        # completar aqui
        # declarar como vector torch y completar con el target correcto. 
        # cuidado las dimensiones
        label = torch.zeros(batch_size, dtype=torch.float, device=device)
        label.fill_(real_label)
        ########################
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        ########################
        # completar aqui
        # declarar como vector torch y completar con el target correcto. 
        # cuidado las dimensiones
        label = torch.zeros(batch_size, dtype=torch.float, device=device)
        label.fill_(fake_label)
        ########################
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        ########################
        # completar aqui
        # declarar como vector torch y completar con el target correcto. 
        # cuidado las dimensiones
        label.fill_(real_label)
        ########################
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

vutils.save_image(real_cpu,
        '%s/real_samples.png' % outf,
        normalize=True)
