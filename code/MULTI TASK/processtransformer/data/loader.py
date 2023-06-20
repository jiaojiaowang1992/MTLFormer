import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing 

from ..constants import Task

class LogsDataLoader:
    def __init__(self, name, dir_path = "./datasets"):
        """Provides support for reading and 
            pre-processing examples from processed logs.
        Args:
            name: str: name of the dataset as used during processing raw logs
            dir_path: str: Path to dataset directory
        """
        self._name = name
        self._dir_path = f"{dir_path}/{name}/processed"

    def prepare_data_next_activity(self, df, 
        x_word_dict, y_word_dict, 
        max_case_length, shuffle=True): #预处理下一个事件任务的数据
        #特征
        x = df["prefix"].values
        y = df["next_act"].values
#        if shuffle:
#            x, y = utils.shuffle(x, y)#打乱顺序

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        # token_x = np.array(token_x, dtype = np.float32)

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[str(_y)])
        # token_y = np.array(token_y, dtype = np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length) #增加维度并标准化

        token_x = np.array(token_x, dtype=np.float32) #数值化
        token_y = np.array(token_y, dtype=np.float32) #数值化

        return token_x, token_y

    def prepare_data_times(self, df,
        x_word_dict, max_case_length,
        time_scaler = None, y_nt_scaler = None,
        y_rt_scaler = None,shuffle = True): #预处理两个时间的任务数据
        #特征
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", 
            "time_passed"]].values.astype(np.float32)
        y_nt = df["next_time"].values.astype(np.float32)
        y_rt = df["remaining_time_days"].values.astype(np.float32)
#        if shuffle:
#            x, time_x, y_nt, y_rt = utils.shuffle(x, time_x, y_nt, y_rt)#打乱顺序

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler() #标准化
            time_x = time_scaler.fit_transform(
                time_x).astype(np.float32) #预处理
        else:
            time_x = time_scaler.transform(
                time_x).astype(np.float32) #预处理

        if y_nt_scaler is None:
            y_nt_scaler = preprocessing.StandardScaler() #标准化
            y_nt = y_nt_scaler.fit_transform(
                y_nt.reshape(-1, 1)).astype(np.float32) #预处理
        else:
            y_nt = y_nt_scaler.transform(
                y_nt.reshape(-1, 1)).astype(np.float32)
        if y_rt_scaler is None:
            y_rt_scaler = preprocessing.StandardScaler() #标准化
            y_rt = y_rt_scaler.fit_transform(
                y_rt.reshape(-1, 1)).astype(np.float32) #预处理
        else:
            y_rt = y_rt_scaler.transform(
                y_rt.reshape(-1, 1)).astype(np.float32) #预处理

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length) #标准化
        
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y_nt = np.array(y_nt, dtype=np.float32)
        y_rt = np.array(y_rt, dtype=np.float32)

        return token_x, time_x, y_nt, y_rt, time_scaler, y_nt_scaler, y_rt_scaler

    def get_max_case_length(self, train_x): #获取长度
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)

    def load_data(self, task): #加载数据
        if task not in (Task.NEXT_ACTIVITY,
            Task.TIMES):
            raise ValueError("Invalid task.")
        #读取训练集和测试集
        train_df = pd.read_csv(f"{self._dir_path}/{task.value}_train_{self._name}.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task.value}_test_{self._name}.csv")
        #打开元数据集
        with open(f"{self._dir_path}/metadata_{self._name}.json", "r") as json_file:
            metadata = json.load(json_file)
        #获取各个参数
        x_word_dict = metadata["x_word_dict"]#特征
        y_word_dict = metadata["y_word_dict"]#标签
        max_case_length = self.get_max_case_length(train_df["prefix"].values)#流程最长长度
        vocab_size = len(x_word_dict) #特征长度
        total_classes = len(y_word_dict)#活动长度

        return (train_df, test_df, 
            x_word_dict, y_word_dict, 
            max_case_length, vocab_size, 
            total_classes)
