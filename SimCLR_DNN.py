import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.l1=nn.Linear(3*75*70, 5000)
        self.l2=nn.Linear(5000, 512)
        self.l3=nn.Linear(512, 256)

    def forward(self, x):
        h=self.l2(self.l1(x.view(-1, 3*75*70)))
        x=F.relu(h)
        x=self.l3(x)
        return h, x 

class DataTransform(object):
    def __init__(self, transform):
        self.transform=transform

    def __call__(self, sample):
        xi=self.transform(sample)
        xj=self.transform(sample)
        return xi, xj
    
class NTXentLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size=batch_size
        self.temperature=temperature
        self.device=device
        self.softmax=torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr=self._get_correlated_mask().type(torch.bool)
        self.similarity_function=self._get_similarity_function()
        self.criterion=torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self):
        self._cosine_similarity=torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _get_correlated_mask(self):
        diag=np.eye(2*self.batch_size)
        l1=np.eye((2*self.batch_size), 2*self.batch_size, k=-self.batch_size)
        l2=np.eye((2*self.batch_size), 2*self.batch_size, k=self.batch_size)
        mask=torch.from_numpy((diag+l1+l2))
        mask=(1-mask).type(torch.bool)
        return mask.to(self.device)

    def _dot_simililarity(x, y):
        v=torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v=self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # print(zis.size()) : torch.Size([4, 256])
        # print(zjs.size()) : torch.Size([4, 256])
        representations=torch.cat([zjs, zis], dim=0) # 8*256
        # print("representations:",representations.size())
        similarity_matrix=self.similarity_function(representations, representations) # 8*8
        #print(similarity_matrix.size())
        l_pos=torch.diag(similarity_matrix, self.batch_size) # torch.Size([4])
        #print(l_pos)
        #print("l_pos:",l_pos.size())
        r_pos=torch.diag(similarity_matrix, -self.batch_size) # torch.Size([4])
        #print(r_pos)
        #print("r_pos:",r_pos.size())
        positives=torch.cat([l_pos, r_pos]).view(2*self.batch_size, 1)
        #print("positives:",positives)
        negatives=similarity_matrix[self.mask_samples_from_same_repr].view(2*self.batch_size, -1)
        #print("negatives:",negatives)
        logits=torch.cat((positives, negatives), dim=1)
        logits/=self.temperature
        labels=torch.zeros(2*self.batch_size).to(self.device).long()
        loss=self.criterion(logits, labels)
        return loss/(2*self.batch_size)
    
batch_size=32
epoch=1000
data_transforms=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize((75,70), interpolation=2),
                                    transforms.ToTensor()])
image_data=torchvision.datasets.ImageFolder(r'C:/Users/DART/Jupyter notebook/med/image_origin/',transform=DataTransform(data_transforms))
train_loader=DataLoader(image_data, batch_size=batch_size, drop_last=True, shuffle=False)
model=DNN()
model.cuda()
optimizer=optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_function=NTXentLoss(device, batch_size, 0.5)

plot_loss=[]
for i in tqdm(range(epoch)):
    running_loss=0
    for (xis, xjs), _ in train_loader:
        # print(xis.size()) : torch.Size([4, 3, 15, 14])
        # print(xjs.size()) : torch.Size([4, 3, 15, 14])
        optimizer.zero_grad()
        ris, zis=model(xis.to(device))  
        rjs, zjs=model(xjs.to(device))  
        zis=F.normalize(zis, dim=1)
        zjs=F.normalize(zjs, dim=1)
        
        loss=loss_function(zis, zjs)
        loss.backward()
        optimizer.step()
        running_loss+=loss
    plot_loss.append(running_loss)

plt.plot(plot_loss)
plt.show()
