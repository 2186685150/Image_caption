import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from keras_preprocessing.sequence import pad_sequences

from Forward_net import Net
from config import max_word_length,text_model_path,model_weights_path,val_features_path,vector_size


    # 加载训练好的模型
if __name__=="__main__":

    image_id="247778426_fd59734130"
    # 加载训练好的模型权重参数
    print("____模型加载ing____")
    model=Net().built_wv_net()
    model.load_weights(model_weights_path)
    print("____模型加载完成！____")
    # 加载提取好的图片特征
    image_features=pickle.load(open(val_features_path,"rb"))
    image_feature=image_features[image_id][0]
    image_feature=np.array(tf.reshape(image_feature,[1,-1]))
    print("____图片特征获取成功____")
    word2vec = Word2Vec.load(text_model_path)
    print("——————————加载语料模型成功————————————")
    # 设置预测容器
    sentence=[]
    word=word2vec.wv["<start>"]
    sentence.append(np.reshape(word,[-1,vector_size]))
    # 填充
    sentences=pad_sequences(sentence,maxlen=max_word_length,dtype=np.float32,padding="post")

    result=[]
    for i in range(1,max_word_length):
        # 开始预测
        pred_wv=model.predict([sentences,image_feature])
        # 解码
        pre_word=word2vec.wv.most_similar(pred_wv,topn=5)
        if pre_word[0][0] !="<end>":
            result.append(pre_word[0][0])
            sentences[0][i]=word2vec.wv[pre_word[0][0]]
        else:
            break
    print(result)