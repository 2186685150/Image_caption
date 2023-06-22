# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : Text_Process.py
# @Software : PyCharm
# @Time : 2023/5/28 9:13
# -----------------------------
"""
加载及预处理语料模块
(Loading And Preprocessing The Text of Image Captions.)
"""
import os
import pickle
import jieba
import re
from gensim.models import Word2Vec

from Config import vector_size, Image_ID_Path, Text_corpus_path, save_word2vec_path, word_type


class Text_loader():
    def __init__(self, size, image_id, text_corpus, words_type):

        self.vector_size = size
        self.image_id_path = image_id
        self.text_corpus_path = text_corpus
        self.train_dict = None
        self.val_dict = None
        self.caption_dict = None
        self.max_length = None
        self.model = None
        self.word_type = words_type

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

    def read(self, usage="train"):
        """
        读取描述文件并根据图片id对应存储
        :return:dict{key:image_id,value:list(len(5)}
        """
        caption_dict = {}
        with open(self.text_corpus_path, "r", encoding="utf-8") as lines:
            for line in lines:
                key = line.strip("\n").split(".jpg")[0]  # strip以换行符进行分割句子，split在.jpg前将句子分割为列表
                value = line.strip("\n").split(" ")[1]
                if key not in caption_dict:
                    caption_dict[key] = []
                caption_dict[key].append(value)
        self.caption_dict = self.token(caption_dict, True)  # 保存读取的全部图片的描述并进行分词。 (Save all images corpus and token)
        if usage == "train":
            image_id_path = self.image_id_path + "flickr8ktrain.txt"
            train_id = self.read_id(image_id_path)
            train_dict = {k: v for k, v in caption_dict.items() if k in train_id}
            print("_______________________________________________________")
            print("...Successfully loaded the pictures captions of train.")
            print("_______________________________________________________")
            self.train_dict = train_dict
        else:
            image_id_path = self.image_id_path + "flickr8ktest.txt"
            val_id = self.read_id(image_id_path)
            val_dict = {k: v for k, v in caption_dict.items() if k in val_id}
            print("_______________________________________________________")
            print("...Successfully loaded the pictures captions of validation.")
            print("_______________________________________________________")
            self.val_dict = val_dict

    def token(self, caption, is_clean=False):
        """
        使用jieba对读取的语料进行分词处理
        :return: type:dict{key:image_id,value:list[5[list]}
        """
        for k, v in caption.items():
            for i in range(len(v)):
                if is_clean:
                    remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
                    v[i] = re.sub(remove_chars, " ", str(v[i]))  # 去除标点符号
                # 进行分词
                v[i] = jieba.lcut(v[i], cut_all=False)  # True为全模式
        print("_______________________________________________________")
        print("...Successfully token the corpus of {} image.".format(len(caption)))
        print("_______________________________________________________")
        return caption

    def bag_word(self):
        """
        使用词袋模型生成词序列。
        (Use a bag-of-words model to generate word sequences.)
        :return: word2idx type :zip{key:word values:idx}  idx2word type: list[len(5551)]
        """

        if not (os.path.isfile("../weights/vocabulary.p")):
            vocabulary = set()
            caption = self.caption_dict.values()
            for sentences in caption:
                for sentence in sentences:
                    for word in sentence:
                        vocabulary.add(word)
            vocabulary.add("<start>")
            vocabulary.add("<end>")
            vocabulary.add("<unknown>")
            # 保存词库
            file_name=r"vocabulary.p"
            with open("../weights/" + file_name, "wb") as f:
                pickle.dump(vocabulary, f)
            print("_______________________________________________________")
            print("...Successfully Saved the bag word model vocabulary.")
            print("_______________________________________________________")
        else:
            vocabulary=pickle.load(open("../weights/vocabulary.p","rb"))
            print("_______________________________________________________")
            print("...Successfully Loaded the model vocabulary.")
            print("_______________________________________________________")
        idx2word = sorted(vocabulary)
        word2idx = dict(zip(idx2word, range(len(vocabulary))))
        return idx2word,word2idx,vocabulary

    def word2vec(self):
        """
        将准备好的语料喂给word2vec转换为对应词向量,相同的语料但词空间可能存在一定差异，所以要保证word2vec模型相同
        (Feed the prepared corpus to word2vec to
        convert it into corresponding word vector. The same corpus may have some differences in word space,
        so make sure the word2vec model is the same.)
        :return: type:tensor(vector_size,)
        """
        if not (os.path.isfile("../model/word2vec_model")):
            caption = self.caption_dict.values()
            sentences = []
            for sentence in caption:
                sentences += sentence
            sentences += [["<start>"], ["<end>"], ["<unknown>"]]
            # 设置训练word2vec模型参数(Set the parameters of model Word2vec)
            model = Word2Vec(sentences, vector_size=self.vector_size, alpha=0.018, min_count=1, window=5, seed=1, sg=1)
            print(model.wv.most_similar("女孩", topn=5))
            print(model.wv.most_similar("玩具", topn=5))
            print(model.wv["女孩", "<start>"].shape)
            # 保存模型
            model.save("../model/word2vec_model")
            print("_______________________________________________________")
            print("...Successfully Saved the model word2vec.")
            print("_______________________________________________________")
        else:
            print("_______________________________________________________")
            print("...Successfully Loaded the model word2vec.")
            print("_______________________________________________________")
            model = Word2Vec.load("../model/word2vec_model")
            print(model.wv.most_similar("女孩", topn=5))
            print(model.wv.most_similar("玩具", topn=5))
        self.model = model

    def max_sentences_lenth(self):
        """
        求出语料句子中长度最长的句子
        (Find the longest sentences in the corpus)
        :return:int
        """
        caption = self.caption_dict
        # 找出所有句子中长度最长的句子
        word_length = []
        for sentences in caption.values():
            word_length.append(len(max(sentences, key=len)))
        max_length = max(word_length)
        print("最长的句子长度为：", max_length)
        self.max_length = max_length

    def data_process(self, usage="train"):
        """
        将文本数据转换为可以输入训练模型的矩阵
        (Convert text data into a matrix that can be fed into the trained model)
        :return: type: list[dict{"image_id","text_wv":list[],"pre":list[]}]
        """
        if usage == "train" and self.train_dict is not None:
            caption = self.train_dict
        elif usage == "val" and self.val_dict is not None:
            caption = self.val_dict
        else:
            return None
        # pad_array=np.zeros((1,self.vector_size))
        featuers = []

        if self.word_type=="wv":
            for k, v in caption.items():
                last_word = "<start>"
                for i in range(len(v)):
                    sentence = []
                    for index, word in enumerate(v[i]):
                        if word not in self.model.wv.key_to_index.keys():
                            word = "<unknown>"
                        sentence.append(self.model.wv[last_word])
                        featuers.append({"image_id": k, "text_wv": list(sentence), "pre": self.model.wv[word]})
                        last_word = word
                    sentence.append(self.model.wv[last_word])
                    featuers.append({"image_id": k, "text_wv": list(sentence), "pre": self.model.wv["<end>"]})
        else:
            idx2word,word2idx,vocab=self.bag_word()
            for k, v in caption.items():
                last_word = "<start>"
                for i in range(len(v)):
                    sentence = []
                    for index, word in enumerate(v[i]):
                        if word not in vocab:
                            word="<unknown>"
                        sentence.append(word2idx[last_word])
                        featuers.append({"image_id": k, "text_wv": list(sentence), "pre":word2idx[word]})
                        last_word = word
                    sentence.append(word2idx[last_word])
                    featuers.append({"image_id": k, "text_wv": list(sentence), "pre": word2idx["<end>"]})

        # 保存数据集 (Save the dataset!)
        if self.word_type=="wv":
            file_name = "model_{}_text_of_{}".format(self.word_type, usage)
        else:
            file_name = "model_{}_text_of_{}".format(self.word_type, usage)

        with open("../weights/" + file_name, "wb") as f:
            pickle.dump(featuers, f)
        print("_______________________________________________________")
        print("...Successfully Processed the {} {} Dataset of Text.".format(self.word_type,usage))
        print("_______________________________________________________")

    def run(self, usage="train"):
        self.read(usage)
        if self.word_type == "wv":
            self.word2vec()
        self.data_process(usage)


if __name__ == "__main__":
    loader = Text_loader(vector_size, Image_ID_Path, Text_corpus_path, word_type[0])
    loader.run("val")  # 默认为train模式
