# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : Image_Loader.py
# @Software : PyCharm
# @Time : 2023/5/29 9:23
# -----------------------------
"""
加载图片并使用CNN网络提取特征。
（Load images and extract features using a CNN network.）
"""
import numpy as np
import os
from tqdm import tqdm
import torch
import tensorflow as tf
from keras.utils import load_img, img_to_array
import pickle

import YOLO
from Config import image_size,Image_ID_Path,Image_path,pre_model_path
class Image_loader():
    def __init__(self):
        self.size =image_size
        self.model=YOLO.YOLO(1,[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    def read_id(self, path):
        """
        读取图片id
        :return:type:list(len(6000 or 1000))(train:val)
        """
        id_data = []
        with open(path, "r", encoding="utf-8") as lines:
            for line in lines:
                data_line = line.strip("\n").split()[0]
                id_data.append(data_line)
        print("...Successfully loaded {} pictures id.".format(len(id_data)))
        return id_data

    def load_pictures(self,id_data,usage="train"):
        """
        读取图片并转换为tensor格式
        (Read pictures and convert to tensor format)
        :return:tensosr(1,3,256,256)
        """
        images=[]
        for i in tqdm(range(len(id_data))):
            image_name = os.path.join(Image_path, id_data[i] + ".jpg")
            image = load_img(image_name, target_size=(self.size, self.size))  # reshape为256
            image = img_to_array(image)
            # 转化为C*H*W格式
            image = image.transpose(2, 0, 1)
            image = image[np.newaxis, :]  # 拓维
            images.append(torch.tensor(image / 255))
        print("_______________________________________________________")
        print("Successfully transformed the {} pictures,and the tensor shape is {}".format(usage,images[0].shape))
        print("_______________________________________________________")

        return images

    def load_weight(self,is_pre=True):
        """
        加载预训练模型权重。
        (Load pretrained model weights.)
        :return: pre_model
        """
        if is_pre:
            if tf.test.is_gpu_available():
                with tf.device("/gpu:0"):
                    weights = torch.load(pre_model_path,map_location="cuda")
            else:
                weights=torch.load(pre_model_path,map_location="cpu")
            model_dict = self.model.state_dict()  # 模型参数字典
            match_dict = {k: v for k, v in weights.items() if k in model_dict}  # 权重参数匹配
            model_dict.update(match_dict)  # 更新模型权重参数
            self.model.load_state_dict(model_dict)  # 更新模型结构
            print("_______________________________________________________")
            print("...Successfully using pre_model parameter states.")
            print("_______________________________________________________")

    def image_features(self,usage="train"):
        """
        使用YOLOv3网络提取图片特征
        (Use the YOLOv3 network to extract image features.)
        :return:dict{key:id values:list(3 tensor(shape:18,8,8 18,16,16 18,32,32))}
        """
        if not os.path.isfile("../weights/{}_features.txt".format(usage)):
            print("_______________________________________________________")
            print("Getting the {} dataset features.".format(usage))
            print("_______________________________________________________")
            self.load_weight()
            if usage=="train":
                id_data=self.read_id(Image_ID_Path+"flickr8ktrain.txt")
            else:
                id_data=self.read_id(Image_ID_Path+"flickr8ktest.txt")
            features={}
            image_tensor=self.load_pictures(id_data,usage)
            for index,data in enumerate(tqdm(image_tensor)):
                with torch.no_grad():
                    feature=list(self.model.forward(data))
                features[id_data[index]] = feature
            pickle.dump(features, open("../weights/{}_features.pkl".format(usage), 'wb'))
            print("_______________________________________________________")
            print("...Successfully Saved the features of {} image".format(usage))
            print("_______________________________________________________")

    def run(self,usage="train"):
        self.image_features(usage)
if __name__=="__main__":
    loader=Image_loader()
    usage="train"
    loader.run(usage="val")