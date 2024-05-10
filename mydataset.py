import os
import one_hot
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
class my_dataset(Dataset):
    def __init__(self,root_dir):
        super(my_dataset,self).__init__()
        self.image_path=[os.path.join(root_dir,image_name)for image_name in os.listdir(root_dir)]
        self.transforms=transforms.Compose(
            [
                transforms.ToTensor(),#将图像数据转换为 PyTorch 张量格式
                transforms.Resize((60,160)),#调整图像尺寸为60*160
                transforms.Grayscale()#转化成灰度图像
            ]
        )
        print(self.image_path)
    def __len__(self):
        return self.image_path.__len__()
    def __getitem__(self, index):
        image_path=self.image_path[index]
        image=self.transforms(Image.open(image_path))#加载图片，并转化成模型可以识别的格式
        label=image_path.split("/")[-1]#将image_path分隔开，只取最后一部分文件名
        label=label.split("_")[0]#将文件名从_分隔开，支取前面一部分，也就是验证码text
        label_tensor=one_hot.text2Vec(label)#4*36，将text转换为一个独热编码向量
        label_tensor=label_tensor.view(1,-1)[0]#1*144
        #label_tensor.view(1,-1)：这部分使用了 PyTorch 中的 view 函数，它的作用是将张量重新塑形成指定的形状。
        # 在这里，1,-1 是参数，表示将原始张量重新塑形成一个行数为 1（即一个行向量），而列数根据原始张量的元素数量自动确定。
        # -1 的作用是让 PyTorch 自动计算这个维度的大小，以保证重塑后的张量总元素数量不变。
        #[0]：这一部分是将经过 view 函数重塑后的张量再次取出。因为 view 函数返回的是一个新的张量，所以通过 [0] 来获取其中的第一个元素，即重塑后的张量。
        return image,label_tensor
if __name__=="__main__":
    writer=SummaryWriter("logs")
    train_data=my_dataset("./datasets/train/")
    img,label=train_data[0]
    print(img.shape,label.shape)
    writer.add_image("img",img,1)
    writer.close()