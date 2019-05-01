from abc import ABC

import torch.nn as nn
import torch
import torch.nn.functional as F


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

    def forward(self, input, hidden=None):
        """

        :param input: shape of [b, s, input_size]
        :param hidden: shape of [num_layers, b, hidden_size]
        :return: - **output** (b, s, hidden_size)
                 - **hidden** (num_layers, b, hidden_size)
        """
        output, hidden = self.gru(input, hidden)
        return output, hidden


class AttnDecoderRNN(nn.Module):

    def __init__(self,
                 hidden_size,
                 input_size,
                 output_size,
                 dropout_p=0.1,
                 num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, encoder_hidden, encoder_outputs):
        """

        :param inputs: of shape [B, S, input_size]
        :param encoder_hidden:  of shape [B, num_layers, hidden_dim]
        :param encoder_outputs: [B, S, hidden_dim]
        :return: - **decoder_outputs**: of shape [max_length, B, output_size]
                 - **decoder_hidden**: of shape [num_layers, B, hidden_dim]
                 - **attn_weights**: of shape [max_length, 1, input_seq_length]
        """

        decoder_input = inputs
        decoder_hidden = encoder_hidden

        # The reason not use Teacher Forcing is: the data type is discrete
        decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

        return decoder_output, decoder_hidden, attn

    def forward_step(self, input, hidden, encoder_outputs):
        """

        :param input: current input of decoder, of shape [batch, in_len, hidden_dim]
        :param hidden: last hidden state of decoder, of shape [num_layers, batch, hidden_dim],
        :param encoder_outputs: hidden state of encoder, of shape [batch, in_seq_len, hidden_dim]
        :return: - **output**: of shape (batch, out_len, out_size)
                 - **hidden**: of shape (num_layers, batch, hidden_dim)
                 _ **attn_weights**: of shape (batch, out_len, in_len)
        """
        input = self.dropout(input)

        output, hidden = self.gru(input, hidden)  # [batch, in_len, hidden_dim], [batch, num_layers, hidden_dim]

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn_weights = torch.bmm(output, encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=2)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # (batch, out_len, hidden_dim)
        combined = torch.cat((attn_applied, output), dim=2)  # (batch, out_len, 2 * hidden_dim)
        output = F.relu(self.attn_combine(combined))  # (batch, out_len, hidden_dim)
        output = self.out(output)  # (batch, out_len, out_size)

        return output, hidden, attn_weights


class WeatherModel(nn.Module):

    def __init__(self, enc_input_size,
                 dec_input_size,
                 hidden_size,
                 num_layers=2,
                 dropout_p=0.2):
        super().__init__()
        self.encoder = EncoderRNN(input_size=enc_input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout_p=dropout_p)
        self.decoder = AttnDecoderRNN(
            input_size=dec_input_size,
            hidden_size=hidden_size,
            output_size=enc_input_size,
            num_layers=num_layers,
            dropout_p=dropout_p)

    def forward(self, input_var, target_var):
        encoder_outputs, encoder_hidden = self.encoder(input_var)
        output, decoder_hidden, attn_weights = self.decoder(
            inputs=target_var,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs)
        return output, decoder_hidden, attn_weights
