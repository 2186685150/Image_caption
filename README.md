# Image_caption
***
## 代码简介
* 这个项目使用的数据集是Flickr8k数据集，每张图片都有五段对应的不同描述（有中文也有英文），项目的最终结果是为了训练一个能够提取图片特征以及语料特征的生成器模型，最终能够对一张图片自动生成描述，这些描述均来自词库。
  在这个项目中仍然有较多未完善的地方，例如模型的CNN部分，我使用的是YOLOv3的作为图片特征提取，结果并没有我预期那么好（原因是没有考虑提取之后多尺度图片特征的空间顺序问题？）以及RNN部分没有很好的一个网络，最终模型效果出现了过拟合。
## 网络结构
* 模型根据语料处理分类：词袋模型，词向量模型两种，经过测试分类模型最终效果优于回归模型。
- 词袋模型结构如下：
![bw_model](https://github.com/2186685150/Image_caption/assets/116703314/30fa4dac-79df-4139-b460-39b3992d9c1c)
-词向量模型如下：
![wv_model](https://github.com/2186685150/Image_caption/assets/116703314/88a0756f-5e30-4011-adff-04197b8387c8)
-可以看到两者之间仅在输入层以及输出层有较大差距。
