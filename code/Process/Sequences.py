# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : Sequences.py
# @Software : PyCharm
# @Time : 2023/5/29 16:27
# -----------------------------
import pickle
import numpy as np
import tensorflow as tf
from keras.utils import Sequence,to_categorical
from keras_preprocessing.sequence import pad_sequences

from Config import vector_size, image_size,input_image_index,batch_szie,input_image_size,word_type,vocabulary_size


class Generator(Sequence):
    def __init__(self,i, usage="train"):
        print("初始化相关参数配置。")  # (Initialize the relevant parameter configuration.)
        self.vector_size = vector_size
        self.image_size = image_size
        self.batch_size = batch_szie
        self.word_type =word_type[i]   # 修改语料类型
        self.text_feature = pickle.load(
            open("../weights/model_{}_text_of_{}".format(self.word_type, usage), "rb"))  # 文本
        self.image_feature = pickle.load(open("../weights/{}_features.pkl".format(usage), "rb"))  # 图片
        print("生成语料数据为：",self.word_type)

    def __getitem__(self, index):
        """
        :param index:
        :return: ndarray[[batchsize,maxlength,vectorsize],[batchsize,imageflatten]],[batchsize,vectorsize]
        """
        # 根据len函数返回值生成并返回对应batch数据
        i = index * self.batch_size

        length = min(self.batch_size, len(self.text_feature) - i)

        # 创建空矩阵
        batch_image_input=np.zeros((length,input_image_size),dtype=np.float32)   # 存放照片特征矩阵
        if self.word_type=="wv":
            batch_pre=np.zeros((length,self.vector_size),dtype=np.float32)   # 存放预测词的词向量矩阵
        else:
            batch_pre=np.zeros((length,vocabulary_size),dtype=np.float32)
        text_input=[]
        # 遍历处理每一张图片以及对应描述
        for i_batch in range(length):
            text_sample = self.text_feature[i + i_batch]
            image_id = text_sample["image_id"]

            image_sample=self.image_feature[image_id][input_image_index]
            # reshape
            image_sample=tf.reshape(image_sample,[1,-1])    # (1152,1)(4068,1)(18432,1)
            batch_image_input[i_batch]=np.array(image_sample)
            if word_type=="wv":
                batch_pre[i_batch]=text_sample["pre"]
            else:
                batch_pre[i_batch]=to_categorical(text_sample["pre"],vocabulary_size)
            text_input.append(text_sample["text_wv"])

        bacth_text_input=pad_sequences(text_input,maxlen=36,padding="post",dtype=np.float32)

        return [bacth_text_input,batch_image_input],batch_pre

    def __len__(self):
        # 返回一个epoch迭代的次数（即Generator输入的数据）
        # print(int(np.ceil(len(self.text_feature)/float(self.batch_size))))
        return int(np.ceil(len(self.text_feature) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.text_feature)    # 打乱顺序

if __name__=="__main__":
    for i in range(2):
        for x,z in enumerate(Generator(1)):
            print(x)