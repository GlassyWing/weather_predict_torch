from abc import ABC

import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module, ABC):

    def __init__(self):
        super().__init__()


class EncoderRNN(nn.Module, ABC):

    def __init__(self, input_size, hidden_size, num_layers=2, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers,
                          dropout=dropout_p,
                          batch_first=True)

    def forward(self, input, hidden):
        """

        :param input: shape of [b, s, input_size]
        :param hidden: shape of [num_layers, b, hidden_size]
        :return: - **output** (b, s, hidden_size)
                 - **hidden** (num_layers, b, hidden_size)
        """
        output, hidden = self.gru(input, hidden)
        return output, hidden


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, num_layers=2, max_length=15):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, encoder_hidden, encoder_outputs):
        """

        :param inputs: shape of [B, S, hidden_dim]
        :param encoder_hidden:  shape of [B, num_layers, hidden_dim]
        :param encoder_outputs: [B, S, hidden_dim]
        :return:
        """

        decoder_outputs = []

        decoder_input = inputs[:, 0].unsqueeze(1)  # [B, hidden_dim]
        decoder_hidden = encoder_hidden
        for di in range(self.max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            decoder_input = decoder_output[:, -1, :]

        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden, encoder_outputs):
        """

        :param input: [batch, hidden_dim]
        :param hidden: [num_layers, batch, hidden_dim]
        :param encoder_outputs: [batch, in_seq_len, hidden_dim]
        :return:
        """
        input = self.dropout(input)

        output, hidden = self.gru(input, hidden)  # [batch, 1, hidden_dim], [batch, num_layers, hidden_dim]

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn_weights = torch.bmm(output, encoder_outputs)
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # (batch, out_len, hidden_dim)
        combined = torch.cat((attn_applied, output), dim=2)  # (batch, out_len, 2 * hidden_dim)
        output = F.relu(self.attn_combine(combined))  # (batch, out_len, hidden_dim)

        return output, hidden, attn_weights


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

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * 2)
        self.fc.bias.data[0] = 1
        self.fc.bias.data[1] = 0.5

    def forward(self, x, position):
        out = self.linear(position)  # [b, s, h]
        out = F.relu(out)
        out = self.fc(out)
        gamma, beta = out.chunk(2, 2)  # [b, s, o]

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
                                output_size=inner_hidden_size,
                                hidden_size=hidden_size)

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
