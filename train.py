import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, optimizer
from mydataset import my_dataset
from mymodel import  mymodel
if __name__=="__main__":
    test_dataset=my_dataset("./datasets/test/")
    test_dataloader=DataLoader(test_dataset,batch_size=40,shuffle=True)

    train_dataset = my_dataset("./datasets/train/")
    train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True)
    loss_fn=nn.MultiLabelSoftMarginLoss().cuda()
    total_step=0
    w=SummaryWriter("logs")
    mymodel=mymodel().cuda()
    optim=Adam(mymodel.parameters(),lr=0.001)
    for epoch in range(10):
        print("外层训练次数{}".format(epoch))
        for i,(images,labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            mymodel.train().cuda()
            output = mymodel(images)
            loss = loss_fn(output, labels)
            optim.zero_grad()  # 把梯度归零
            loss.backward()  # 反向传播
            optim.step()
            if i % 100 == 0:
                total_step += 1
                print("训练{}次,损失率{}".format(total_step*100,loss.item()))
                w.add_scalar("loss",loss.item(),total_step)
    w.close()
        #enumerate() 函数用于同时获取数据以及它们在批次中的索引
        # 其中 i 是索引号，(images, labels) 则是对应的图像数据和标签。
torch.save(mymodel,"model.pth")