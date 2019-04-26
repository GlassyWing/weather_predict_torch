from abc import ABC

import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):

    def __init__(self, input_size, hidden_size, inner_hidden_size, max_len=15, num_layers=2, dropout=0.2):
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
        self.linear = nn.Linear(inner_hidden_size, input_size)

    def forward(self, x):
        lstm_output, _ = self.lstm1(x)
        lstm_output = self.drop(lstm_output)
        attn_weights = F.softmax(self.attn(lstm_output), dim=2)  # (b,s,s)
        attn_applied = torch.bmm(attn_weights, lstm_output)  # (b,s,h)
        lstm_output, _ = self.lstm2(attn_applied)  # (b,s,h_2)
        out = self.linear(lstm_output)  # (b,s,o)
        return out, attn_weights


class AdaptiveInstanceNorm(nn.Module, ABC):

    def __init__(self,addition_size, hidden_size, seq_len=15, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=addition_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)
        self.linear.bias.data[0] = 1
        self.linear.bias.data[1] = 0

    def forward(self, x, addition):
        lstm_output, _ = self.lstm(addition)
        gamma, beta = self.linear(lstm_output).chunk(2, 2)  # [b, s, 1]
        # x = self.norm(x)
        return x * gamma + beta


class WeatherModel(nn.Module, ABC):

    def __init__(self,
                 addition_size,
                 input_size,
                 hidden_size,
                 inner_hidden_size,
                 max_len=15,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()

        self.base_model = BaseModel(input_size=input_size,
                                    hidden_size=hidden_size,
                                    inner_hidden_size=inner_hidden_size,
                                    max_len=max_len,
                                    dropout=dropout,
                                    num_layers=num_layers)

        self.adain = AdaptiveInstanceNorm(
                                          addition_size=addition_size,
                                          hidden_size=hidden_size,
                                          seq_len=max_len,
                                          num_layers=2,
                                          dropout=dropout)

    def forward(self, input, addition):
        """

        :param input: shape of [b, s, input_size]
        :param addition: shape of [b, s, addition_size]
        :return:
        """

        adain_output = self.adain(input, addition)
        out, attn_weights = self.base_model(adain_output)  # [b, s, input_size]

        return out, attn_weights
