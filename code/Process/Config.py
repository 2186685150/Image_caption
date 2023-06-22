# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : Config.py
# @Software : PyCharm
# @Time : 2023/5/28 9:24
# -----------------------------
"""
用于调整参数以及设置路径模块
(Used to Adjust Parameters And Set Paths)
"""
Image_ID_Path=r"../Flickr8k and Flickr8kCN/flickr8kcn/data/"   # 设置数据集图片id的路径  (Set the path of dataset image id )
Image_path=r"../Flickr8k and Flickr8kCN/Flicker8k_Dataset/"
Text_corpus_path=r"../Flickr8k and Flickr8kCN/flickr8kcn/data/flickr8kzhc.caption.txt"   # 语料文件路径 (The Path of Caption Files)
save_word2vec_path=r"../weights/word2vec_model"
pre_model_path=r"../weights/yolo_weights.pth"
checkpoint_path=r"../model/"
load_model_path=r"../model/"   # 加载模型的文件路径
tensorboard_log_path=r"../logs/"
load_model_name=r"bw_model.0050--3.9706.hdf5"

word_type=["wv",'bw'] # 设置词袋模型还是词向量模型 (Set whether the bag-of-word model or the word-vector model is made)
vector_size=32  # 设置word2vec的词向量长度(Set the word vector length of word2vec)
image_size=256  # 设置图片矩阵的大小 (Sets the size of the picture matrix)
input_image_size=1152   # 对应1152，4068，18432 尺寸flatten之后的图片特征矩阵
input_image_index=0 # 对应0,1,2
# 网络配置参数    (Network configuration parameters)
batch_szie=128
max_sentences_length=36 # 根据个人语料不同调整
vocabulary_size=5551    # 根据个人语料调整
embedding_size=64   # embedding层输出大小
patience=50 # 控制保存模型频次的参数
