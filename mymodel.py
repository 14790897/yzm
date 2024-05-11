import torch
from torch import nn
import common
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#[1,64,30,80](输入的数量，通道数，高，宽)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#[1,128,15,40]
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#[1,256,7,20]
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#[1, 512, 3, 10]
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),#[1, 15360]
            nn.Linear(in_features=15360, out_features=4056),
            nn.Dropout(0.2),
            nn.Linear(in_features=4056, out_features=common.captcha_size*common.captcha_array.__len__())
        )#[1, 144]
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        return x
if __name__=="__main__":
    data=torch.ones(1,1,60,160)
    m=mymodel()
    x=m(data)
    print(x.shape)