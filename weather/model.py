from abc import ABC

import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):

    def __init__(self, input_size,
                 hidden_size, inner_hidden_size,
                 max_len=15, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=inner_hidden_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)
        self.attn = nn.Linear(hidden_size, max_len)

    def forward(self, x):
        lstm_output, _ = self.lstm1(x)
        lstm_output = self.drop(lstm_output)
        attn_weights = F.softmax(self.attn(lstm_output), dim=2)  # (b,s,s)
        attn_applied = torch.bmm(attn_weights, lstm_output)  # (b,s,h)
        lstm_output, _ = self.lstm2(attn_applied)  # (b,s,h_2)
        return lstm_output, attn_weights


class ScaleApply(nn.Module, ABC):
    """依据当前的时间、地点或其它特征对数据进行正规化"""

    def __init__(self, input_size, hidden_size, seq_len=15):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 2)
        self.fc.bias.data[0] = 1
        self.fc.bias.data[1] = 0.5

    def forward(self, x, position):
        out = self.linear(position)  # [b, s, h]
        out = F.relu(out)
        out = self.fc(out)
        gamma, beta = out.chunk(2, 2)  # [b, s, 1]

        return x * gamma + beta


class WeatherModel(nn.Module, ABC):

    def __init__(self,
                 position_size,
                 input_size,
                 hidden_size,
                 inner_hidden_size,
                 max_len=15,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()

        self.wave_model = BaseModel(input_size=input_size + position_size,
                                    hidden_size=hidden_size,
                                    inner_hidden_size=inner_hidden_size,
                                    max_len=max_len,
                                    dropout=dropout,
                                    num_layers=num_layers)

        self.position_norm = nn.BatchNorm1d(max_len)

        self.scale = ScaleApply(input_size=position_size,
                                hidden_size=hidden_size,
                                seq_len=max_len)

        self.fc = nn.Linear(inner_hidden_size, input_size)

    def forward(self, x, position):
        """

        :param x: shape of [b, s, input_size]
        :param position: shape of [b, s, addition_size]
        :return:
        """

        position = self.position_norm(position)
        input = torch.cat((x, position), dim=2)

        wave_out, attn_weights = self.wave_model(input)
        scale_out = self.scale(wave_out, position)

        cross = scale_out + wave_out
        out = self.fc(cross)

        return out, attn_weights
