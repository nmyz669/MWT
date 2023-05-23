import os
import json
import glob

import torch
import matplotlib.pyplot as plt

import numpy as np
from MWT import MWT as create_model
import cv2
from augmentations import *



def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    # plt.show()


def predict():
    im_height = 224
    im_width = 224
    num_classes = 2
    batch_size = 32# 每次预测时将多少张图片打包成一个batch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load images
    # 指向需要遍历预测的图像文件夹
    root = "/mnt/disk1/yjx_data/class2/data8"

    # root = "H:/large_data/large_data/data_5"
    neg_dir = root + "/neg/"  # 训练集路径
    pos_dir = root + "/pos/"  # 验证集路径

    assert os.path.exists(neg_dir), f"file: '{neg_dir}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    neg_path_list = [os.path.join(neg_dir, i) for i in os.listdir(neg_dir) if i.endswith(".jpg")]

    assert os.path.exists(pos_dir), f"file: '{pos_dir}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    pos_path_list = [os.path.join(pos_dir, i) for i in os.listdir(pos_dir) if i.endswith(".jpg")]

    # create model
    model = create_model(num_classes=2).to(device)

    # load model weights
    weights_path = "./M15/model-98.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    test_pre_labels2 = []
    test_pre_labels1 = []
    heat_maps = np.zeros((2, 2))
    # prediction
    model.eval()
    with torch.no_grad():
        for ids in range(0, len(neg_path_list) // batch_size):
            img_list = []
            for img_path in neg_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                # img = Preprocess()(img.copy())
                img = Augmentations.Normalization((0, 1))(img)
                img = np.array(img, np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)
                img = torch.Tensor(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                test_pre_labels1.append(cla.numpy())

    with torch.no_grad():
        for ids in range(0, len(pos_path_list) // batch_size):
            img_list = []
            for img_path in pos_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                # img = Preprocess()(img.copy())
                img = Augmentations.Normalization((0, 1))(img)
                img = np.array(img, np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)
                img = torch.Tensor(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                test_pre_labels2.append(cla.numpy())

    for test_pre_label in zip(test_pre_labels1):
        heat_maps[0][test_pre_label] = heat_maps[0][test_pre_label] + 1

    for test_pre_label in zip(test_pre_labels2):
        heat_maps[1][test_pre_label] = heat_maps[1][test_pre_label] + 1

    class_names = ['neg', 'pos']
    print(heat_maps)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps,
                  save_name="results/test8_98_M15.png")


if __name__ == '__main__':
    predict()

