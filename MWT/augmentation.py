from augmentations import Augmentations
import cv2
import numpy as np

class Preprocess(object):
    """
    测试成功： 20220427 bruce
    ！！！使用call方法原图将会改变 ！！！
    example:
    img = Preprocess()(img) # img改变
    img2 = Preprocess()(img.copy()) # img不变
    """

    def __init__(
            self,
            sharp=0.5,
            blur=0.5,
            hsv_disturb=0.5,
            rgb_switch=0.5,
            rotate=0.5,
            h_flip=0.5,
            v_flip=0.5,
            norm=(0, 1),
    ):
        self.pre_funcs = [
            Augmentations.RandomSharp(sharp),
            Augmentations.RandomGaussainBlur(blur),
            Augmentations.RandomHSVDisturb(hsv_disturb),
            Augmentations.RandomRGBSwitch(rgb_switch),
            Augmentations.RandomRotate90(rotate),
            Augmentations.RandomHorizontalFlip(h_flip),
            Augmentations.RandomVerticalFlip(v_flip),
        ]
        self.preprocess = Augmentations.Compose(*self.pre_funcs)
        self.normalize = Augmentations.Normalization(norm)

    def __call__(self, image):
        image = self.preprocess(image)
        image = self.normalize(image)
        return image


# img = cv2.imread("H:/DL/bishe/code1/test/neg-test/test_305.jpg")
# print(type(img))
# print(img)

# # img3 = cv2.imread("H:/DL/bishe/code1/test/neg-test/test_345.jpg")
# img = Preprocess()(img)
# cv2.imshow('img',img)
# img2 = Preprocess()(img.copy())
# print(type(img2))
# print(img2)

# # imgs = np.hstack([img, img3])
# # cv2.imshow('imgs', img2)
# # # cv2.imshow('img',img)
# cv2.waitKey(0)
