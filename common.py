import os
import random
import time

captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size = 4
from captcha.image import ImageCaptcha

if __name__ == "__main__":
    for i in range(1000):
        image = ImageCaptcha()
        image_text = "".join(random.sample(captcha_array, captcha_size))
        image_path = "./datasets/train/{}_{}.png".format(image_text, time.time())
        print(image_path)
        image.write(image_text, image_path)
