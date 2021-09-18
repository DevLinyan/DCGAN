import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataLoader import dataset
from models.DCGAN import Discriminator, Generative

from torch.optim import Adam
import cv2

config = {
    'batch_size': 128,
    'epoch': 5
}

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    # models
    Dis = Discriminator()
    Gen = Generative()
    # dataset
    dataloader = DataLoader(dataset, config['batch_size'], shuffle=True, num_workers=2)
    # criterion
    criterion = nn.BCELoss()
    # ptimizer
    optimizer_G = Adam(Gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = Adam(Dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # iterative
    loss_Gen = 0
    loss_Dis = 0
    for epoch in range(config['epoch']):
        for i, data in enumerate(dataloader):
            Dis.zero_grad()
            Dis.to(device)

            label = torch.full((data[0].size(0),), 1, dtype=torch.float32)
            output = Dis(data[0].to(device)).view(-1).cpu()
            Disloss1 = criterion(output, label)
            Dx = Disloss1.mean().item()
            Disloss1.backward()

            noise = torch.randn((data[0].size(0), 100, 1, 1), device=device)
            Gen.to(device)
            fake = Gen(noise)
            output = Dis(fake.detach()).view(-1)
            fakelabel = torch.full((data[0].size(0),), 0, dtype=torch.float32, device=device)
            Disloss2 = criterion(output, fakelabel).cpu()
            D_G_1 = Disloss2.mean().item()
            Disloss2.backward()
            DiscriminatorLoss =Disloss1.item() + Disloss2.item()

            optimizer_D.step()

            Gen.zero_grad()
            Gen.to(device)

            label = torch.full((data[0].size(0), ), 1, dtype=torch.float32, device=device)
            output = Dis(fake).view(-1)
            Genloss = criterion(output, label).cpu()
            D_G_2 = Genloss.mean().item()
            Genloss.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"epoch:{epoch}, i:{i}. Dis_Gen_1 error:{DiscriminatorLoss}, Dis_Gen_2 error:{Genloss.item()}")
    torch.save(Gen, 'models.pth')
    with torch.no_grad():
        Gen.eval()
        for i in range(1):
            noise = torch.randn((10, 100, 1, 1), device=device)
            res = Gen(noise).cpu()
            for j in range(res.shape[0]):
                img = res[0].numpy().transpose(1,2,0)
                cv2.imwrite(f'{j}.jpg', img)












