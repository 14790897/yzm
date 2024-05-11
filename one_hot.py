import torch
import common
def text2Vec(text):#把文本转化成变量
    #4列36行
    vec=torch.zeros(common.captcha_size,len(common.captcha_array))
    for i in range(len(text)):
        vec[i,common.captcha_array.index(text[i])]=1
    return vec
def vec2Text(vec):
    vec=torch.argmax(vec,dim=1)
    text=""
    for i in vec:
        text+=common.captcha_array[i]
    return text
if __name__=="__main__":
    vec=text2Vec("aab1")
    vec=vec.view(1,-1)[0]
    print(vec,vec.shape)
    #print(vec2Text(vec))