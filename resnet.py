import torch
import torch.nn as nn
import  torchvision.models as models

import common


class myresnet(nn.Module):
    def __init__(self):
        super(myresnet,self).__init__()
        self.model=models.resnet50(pretrained=False)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc=nn.Linear(in_features=2048, out_features=common.captcha_size*common.captcha_array.__len__())

    def forward(self, x):
        x=self.model(x)
        return x
if __name__ == '__main__':
    m=myresnet()
    x=torch.randn(1,1,60,160)
    y=m(x)