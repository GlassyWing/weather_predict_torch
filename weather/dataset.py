from abc import ABC
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import pandas as pd
import torch
from math import ceil
from torch.utils.data import Dataset


class Standardization:
    """标准化"""

    def __init__(self, mean, std, epsilon=1e-7):
        self.mean = np.asarray(mean, dtype='float32')
        self.std = np.asarray(std, dtype='float32')
        self.epsilon = epsilon

    def __call__(self, x):
        return ((x - self.mean) / (self.std + self.epsilon)).astype('float32')

    def I(self):
        return StandardizationInverse(self.mean, self.std, self.epsilon)


class StandardizationInverse(Standardization):

    def __init__(self, mean, std, epsilon):
        super().__init__(mean, std, epsilon)

    def __call__(self, x):
        return (x * (self.std + self.epsilon) + self.mean).astype('float32')


class Normalization:
    """
    归一化
    """

    def __init__(self, min, max, epsilon=1e-7):
        self.min = np.asarray(min, dtype='float32')
        self.max = np.asarray(max, dtype='float32')
        self.epsilon = epsilon

    def __call__(self, x):
        return ((x - self.min) / (self.max - self.min + self.epsilon)).astype('float32')

    def I(self):
        return NormalizationInverse(self.min, self.max, self.epsilon)


class NormalizationInverse(Normalization):

    def __init__(self, min, max, epsilon=1e-7):
        super().__init__(min, max, epsilon)

    def __call__(self, x):
        return (x * (self.max - self.min + self.epsilon) + self.min).astype('float32')


def get_place_dict(places_dict_path):
    index2place = []
    with open(places_dict_path, mode="r", encoding='utf-8') as f:
        for s in f:
            index2place.append(s[:-1])
    place2index = {place: i for i, place in enumerate(index2place)}
    return index2place, place2index


def mapping_place(dataset, place2index, mapping_cols=None):
    if mapping_cols is None:
        mapping_cols = ['province', 'city', 'county']
    dataset = pd.read_csv(dataset) if type(dataset) == str else dataset

    dataset['place'] = reduce(lambda col1, col2: col1 + col2, [dataset[col] for col in mapping_cols])
    dataset['place'] = dataset['place'].map(place2index)
    dataset.drop(columns=mapping_cols, inplace=True)
    return dataset


class WeatherDataset(Dataset, ABC):

    def __init__(self, weather_file_path,
                 places_dict_path,
                 src_seq_len=15,
                 tgt_seq_len=15,
                 according=None,
                 conclusion=None,
                 is_reverse=False,
                 transform=None,
                 rolling=None):
        """
        初始化气象数据集
        :param weather_file_path: 气象数据文件路径
        :param places_dict_path: 地点字典
        :param src_seq_len:     原始序列长度
        :param tgt_seq_len:     目标序列长度
        :param according:  根据列
        :param conclusion: 结论列
        :param is_reverse: 指定载入的数据集是否为逆序
        :param transform: 数据变换
        :param rolling: 滑动平均窗口大小
        """
        self.transform = transform
        self.requires = ['place',
                         'year',
                         'month',
                         'day',
                         '24_hours_avg_tem', '24_hours_max_tem',
                         '24_hours_min_tem', '24_hours_avg_pre',
                         '24_hours_avg_rhu', '24_hours_avg_gst',
                         '24_hours_avg_gst_15cm', '24_hours_avg_gst_20cm',
                         ]
        if according is None:
            self.according = ['place',
                              'month',
                              'day',
                              '24_hours_avg_tem', '24_hours_max_tem',
                              '24_hours_min_tem', '24_hours_avg_pre',
                              '24_hours_avg_rhu', '24_hours_avg_gst',
                              '24_hours_avg_gst_15cm', '24_hours_avg_gst_20cm',
                              ]
        else:
            self.according = according
        if conclusion is None:
            self.conclusion = [
                '24_hours_avg_tem', '24_hours_max_tem',
                '24_hours_min_tem', '24_hours_avg_pre',
                '24_hours_avg_rhu', '24_hours_avg_gst',
                '24_hours_avg_gst_15cm', '24_hours_avg_gst_20cm'
            ]
        else:
            self.conclusion = conclusion

        self.acc_sub_con = list(set(self.according) - set(self.conclusion))
        self.acc_sub_con.sort()

        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.index2place, self.place2index = get_place_dict(places_dict_path)
        self.weather = mapping_place(weather_file_path, place2index=self.place2index)
        self.__check_data(self.weather)

        # 如果是逆序，转变为顺序
        if is_reverse:
            self.weather = self.weather[::-1]

        # 进行数据变换
        if self.transform is not None:
            self.weather[self.according] = self.transform(self.weather[self.according])

        self.places = self.weather['place'].unique().tolist()
        self.num_place = len(self.places)
        self.weathers = self.__split_by_place()

        # 执行滑动平均
        if rolling is not None and rolling >= 2:
            for i, weather in enumerate(self.weathers, 0):
                weather[self.conclusion] = weather[self.conclusion].copy().rolling(rolling).mean()
                self.weathers[i] = weather[rolling:]

        self.__breakpoint = self.__weathers_breakpoint()

    def calculate_size(self):
        total_size = 0
        for i in range(len(self.weathers)):
            size_i = len(self.weathers[i]) - self.src_seq_len - self.tgt_seq_len
            total_size += size_i
        return total_size

    def __split_by_place(self):
        return [self.weather[self.weather['place'] == place].copy() for place in self.places]

    def __weathers_breakpoint(self):
        init = [0]
        for i in range(1, len(self.weathers)):
            init.append(init[i - 1] + len(self.weathers[i]) - self.src_seq_len - self.tgt_seq_len)
        return init

    def __getitem__(self, item):

        # 找到下一个断点的位置
        idx = np.where(item < np.array(self.__breakpoint))[0]

        # 获得上一个断点的位置
        if len(idx) == 0:  # 如果已经没有下一个断点
            idx = -1
        else:  # 减一即为上一个断点
            idx = idx[0] - 1

        offset = item - self.__breakpoint[idx]
        data = self.weathers[idx + 1][offset: offset + self.src_seq_len + self.tgt_seq_len]

        x = data[self.conclusion][:self.src_seq_len].values
        y = data[self.conclusion][self.src_seq_len:].values
        additional = data[self.acc_sub_con][self.src_seq_len:].values
        # additional = np.zeros_like(y)

        return {
            'input': torch.from_numpy(x),
            'target': torch.from_numpy(y),
            'additional': torch.from_numpy(additional)
        }

    def __len__(self):
        return self.calculate_size()

    def __check_data(self, weather):
        if len(set(self.requires) - set(weather.columns.tolist())) != 0:
            raise ValueError("数据中必须包含以下列：" + str(self.requires))

    def retain_columns(self):
        return set(self.requires) - set(self.conclusion)


class PreWeatherDataset(WeatherDataset):
    """用于预测的气象数据"""

    def __init__(self, weather_file_path, places_dict_path, **kwargs):
        super().__init__(weather_file_path, places_dict_path, **kwargs)

        if self.num_place != 1:
            raise ValueError("所有气象数据的地点应相同！")

        self.weather = self.weathers[0]
        self.weather = self.weather[: self.src_seq_len]

        dec_input = np.zeros(shape=(self.tgt_seq_len, 4))

        place = self.get_place()
        next_date = self.get_curr_date()

        for i in range(self.tgt_seq_len):
            next_date = next_date + timedelta(days=1)
            dec_input[i, :] = np.array([place, next_date.year, next_date.month, next_date.day], dtype='float32')

        self.future_position = pd.DataFrame(dec_input, columns=['place', 'year', 'month', 'day'])

    def get_place(self):
        return self.weather['place'].iloc[-1]

    def get_curr_date(self):
        curr_date = datetime.strptime(
            str(ceil(self.weather['year'].iloc[-1])) +
            '/' + str(ceil(self.weather['month'].iloc[-1])) +
            '/' + str(ceil(self.weather['day'].iloc[-1])),
            '%Y/%m/%d')
        return curr_date

    def append(self, values):
        """添加数据"""
        future_data = pd.DataFrame(data=values, columns=self.conclusion)
        future_data = pd.concat((self.future_position, future_data), axis=1, sort=False)
        self.weather = pd.concat((self.weather, future_data), axis=0, sort=False)

        if self.transform is not None:
            self.weather[self.according] = self.transform.I()(self.weather[self.according])

    def __getitem__(self, item):
        enc_input = self.weather[self.conclusion].values
        # dec_input = np.zeros(shape=(self.tgt_seq_len, enc_input.shape[1]))
        dec_input = self.future_position[self.acc_sub_con].values
        return {
            "input": torch.from_numpy(enc_input),
            "addition": torch.from_numpy(dec_input)
        }

    def __len__(self):
        return 1
