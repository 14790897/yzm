from PIL import Image
from torch.utils.data import DataLoader
import one_hot
import torch
import common
from resnet import myresnet
from mydataset import my_dataset
from torchvision import transforms


def test_predict():
    test_dataset = my_dataset("./datasets/test/")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    m = torch.load("resnet.pth").cuda()
    print(m)
    test_len = test_dataset.__len__()
    correct = 0
    print(test_len)
    for i, (images, labels) in enumerate(test_dataloader):
        images = images.cuda()
        labels = labels.cuda()
        # print(labels.shape)#[4,144]                                         
        labels = labels.view(-1, common.captcha_array.__len__())
        # print(labels.shape)#[160, 36]                                       
        label_text = one_hot.vec2Text(labels)
        output = m(images)
        output = output.view(-1, common.captcha_array.__len__())
        output_test = one_hot.vec2Text(output)
        # print(output.shape,output_test)                                     
        if label_text == output_test:
            correct += 1
            print("正确值{},预测值{}".format(label_text, output_test))
        else:
            print("正确值{},预测失败值{}".format(label_text, output_test))
    print("正确率{}".format(correct / test_len * 100))


if __name__ == "__main__":
    test_predict()