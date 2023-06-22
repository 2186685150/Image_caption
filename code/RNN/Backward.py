# python3
# -- coding: utf-8 --
# -------------------------------
# @Author : Mr.Akiy
# -------------------------------
# @File : Backward.py
# @Software : PyCharm
# @Time : 2023/5/31 12:56
# -----------------------------
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Forward_net import Net
import keras
import tensorflow as tf
from untils import get_available_gpus

from Sequences import Generator
from Config import patience, checkpoint_path, load_model_path,tensorboard_log_path, word_type,load_model_name

if __name__ == "__main__":

    # 选择并设置网络类型 (Select and set the network type)
    model_type = word_type[1]  # 0 or 1

    tensor_board = keras.callbacks.TensorBoard(log_dir=tensorboard_log_path, histogram_freq=0
                                               , write_graph=True, write_images=True)
    model_save_name = checkpoint_path + "bw_model.{epoch:04d}--{val_loss:.4f}.hdf5"
    model_checkpoint = ModelCheckpoint(model_save_name, monitor="val_loss", verbose=0, save_best_only=True,
                                       mode="min", period=50)  # 50个epoch保存一次文件
    # 设置early_stop
    early_stop = EarlyStopping(monitor="val_loss", patience=200)
    # 自适应调整学习率
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=int(patience/2), verbose=0)


    # 继承库中的函数，重写自己的track方法
    class Mycbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            # 定义需要保存的模型
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            # 定义在每个epoch结束之后都调用
            Format = model_save_name
            self.model_to_save.save(Format % (epoch, logs["val_loss"]))


    # 使用GPU进行训练
    num_gpu = len(get_available_gpus())
    if num_gpu >= 1:
        print("_______________________________________________________")
        print("The device has a total of {} GPUs,The GPU will be used for training".format(num_gpu))
        print("_______________________________________________________")
        with tf.device("/gpu:0"):
            if model_type=="bw":
                new_model = Net().built_bw_net()
            if model_type=="wv":
                new_model = Net().built_wv_net()
            if load_model_path is not None:
                new_model.load_weights(load_model_path+load_model_name)
                print("_______________________________________________________")
                print("...Successfully Loaded the already trained {} model weights file {}".format(model_type,load_model_name))
                print("_______________________________________________________")
    else:
        print("_______________________________________________________")
        print("Your computer doesn't have a GPU available, and the next step will be to use the CPU for training.")
        print("_______________________________________________________")
        if model_type == "bw":
            new_model = Net().built_bw_net()
        else:
            new_model = Net().built_wv_net()
        if load_model_path is not None:
            new_model.load_weights(load_model_path+load_model_name)

    # 定义模型优化器
    adam=keras.optimizers.Adam(learning_rate=5e-4)
    if model_type=="wv":
        new_model.compile(optimizer=adam,loss="mse",metrics=["accuracy"])
    else:
        new_model.compile(optimizer=adam,loss="categorical_crossentropy", metrics=['accuracy'])

    # 实例化generator
    if model_type=="wv":
        train_generator=Generator(0,"train")    # 0=wv 1=bw
        val_generator=Generator(0,"val")
    else:
        train_generator = Generator(1, "train")  # 0=wv 1=bw
        val_generator = Generator(1, "val")
    # 调用callbacks函数
    callback=[tensor_board,model_checkpoint,early_stop,reduce_lr]
    # 进行模型训练
    new_model.fit_generator(train_generator,validation_data=val_generator,
                            epochs=1000,callbacks=callback)
