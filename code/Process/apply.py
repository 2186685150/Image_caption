# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : apply.py
# @Software : PyCharm
# @Time : 2023/6/2 15:28
# -----------------------------
"""
对训练好的词袋模型使用图片进行测试。
(Test the trained bag-of-words model with images.)
"""
import pickle
import os
import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences


from Forward_net import Net
from Text_Process import Text_loader
from Config import load_model_path,load_model_name,vector_size,max_sentences_length,Image_ID_Path,Text_corpus_path,word_type

image_id="247778426_fd59734130"
print("______________________________________")
print("Loaded the model parameters .")
print("______________________________________")
model=Net().built_bw_net()
model.load_weights(load_model_path+load_model_name)
print("...Successfully loaded the weights of model.")
print("Loading the image features.")
image_features=pickle.load(open("../weights/train_features.pkl","rb"))
image_feature = image_features[image_id][0]
image_feature = np.array(tf.reshape(image_feature, [1, -1]))
print("...Successfully loaded the features of image.")
loader=Text_loader(vector_size,Image_ID_Path,Text_corpus_path,word_type[1])
idx2word,word2idx,vocabulary=loader.bag_word()

def prediction(word2idx,idx2word):
    start=[word2idx["<start>"]]
    start_word=[[start,0.0]]

    while len(start_word[0][0])<max_sentences_length:
        temp=[]
        for s in start_word:
            pad=pad_sequences([s[0]],maxlen=36,padding="post",dtype=np.int32)
            pre=model.predict([pad,image_feature])
            word_preds=np.argsort(pre[0])[-3:]

            for word in word_preds:
                next_cap,prob=s[0][:],s[1]
                next_cap.append(word)
                prob+=pre[0][word]
                temp.append([next_cap,prob])

        start_word=temp
        start_word=sorted(start_word,reverse=False,key=lambda x:x[1])
        start_word=start_word[-3:]
    start_word=start_word[-1][0]
    caption=[idx2word[i] for i in start_word]

    captions=[]
    for i in caption:
        if i!="<end>":
            captions.append(i)
        else:
            break
    captions="".join(captions[1:])
    return captions

caption=prediction(word2idx,idx2word)
print(caption)