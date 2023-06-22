# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : Forward_net.py
# @Software : PyCharm
# @Time : 2023/5/29 16:14
# -----------------------------

import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.layers import Embedding, RepeatVector, Concatenate, Dropout, Dense, Input, TimeDistributed, Dot, Activation, \
    GRU
from keras.models import Model

from Config import max_sentences_length, vector_size, input_image_size, vocabulary_size, embedding_size


class Net():

    def built_bw_net(self):
        """
        构造词序列网络模型结构；
        (construct word sequence network model structure)
        :return: bw_model
        """
        print("Using bw_model to train now.")
        print("_______________________________________________________")
        # print("使用双层GRU网络进行分类任务预测。")
        # 定义输入层 (Define the text input layer)
        Text_input = Input(shape=(max_sentences_length,), dtype="int32")
        text_x = Embedding(vocabulary_size, embedding_size)(Text_input)
        text_x = GRU(256,return_sequences=True)(text_x)  # 输出维度为256,输出为所有隐藏层的输出h(t)
        text_x = TimeDistributed(Dense(embedding_size))(text_x)  # 对时间序列数据进行时间步戳处理，共享权重，同时进行dense操作

        # 定义图片输入层(Defines the picture input layer)
        Image_input = Input(input_image_size, )
        image_x = Dense(embedding_size, activation="relu")(Image_input)  # 图片信息进行embedding
        image_x = RepeatVector(1)(image_x)  # 设置时间步戳为1,即每个时间步戳都只有一张图片特征

        # 维度拼接(Dimensional stitching)
        x = [text_x, image_x]
        x = Concatenate(axis=1)(x)  # 在轴axis 1上进行拼接

        x=Dropout(0.1)(x)
        x=GRU(512,name="GRU_1",return_sequences=True)(x)
        x=Dropout(0.2)(x)

        x=GRU(512,name="GRU_2")(x) # 返回最终时间步骤的结果
        x = Dropout(0.2)(x)
        # x = LSTM(256, activation="relu")(x)
        # x = Dropout(0.2)(x)

        x=Dense(256,activation="relu")(x)
        output = Dense(vocabulary_size, activation="softmax")(x)
        inputs = [Text_input, Image_input]
        bw_model = Model(inputs=inputs, outputs=output)

        return bw_model

    def built_wv_net(self):
        # 定义文本输入层
        print("Using wv_model to train now.")
        print("_______________________________________________________")
        Text_input = Input((max_sentences_length, vector_size), dtype="float32")
        text_x = GRU(256, return_sequences=True)(Text_input)
        text_x = TimeDistributed(Dense(embedding_size))(text_x)

        # 定义图片输入层
        Image_input = Input(input_image_size, )
        image_x = Dense(embedding_size, activation="relu")(Image_input)  # 图片信息进行embedding
        image_x = RepeatVector(1)(image_x)

        x = [text_x, image_x]
        x = Concatenate(axis=1)(x)  # 在轴axis 1上进行拼接

        x = Dropout(0.2)(x)
        x = GRU(512, name="GRU_1", return_sequences=True)(x)
        x = Dropout(0.2)(x)

        x = GRU(512, name="GRU_2")(x)  # 返回最终时间步骤的结果
        x = Dropout(0.4)(x)

        output = Dense(vector_size, activation="linear")(x)
        inputs = [Text_input, Image_input]
        wv_model = Model(inputs=inputs, outputs=output)

        return wv_model


if __name__ == "__main__":
    # 使用cpu运行模型
    with tf.device("/cpu:0"):
        model = Net().built_wv_net()
    # 打印模型信息
    print(model.summary())
    # 打印模型结构
    plot_model(model, to_file='wv_model.png', show_layer_names=True, show_shapes=True)
