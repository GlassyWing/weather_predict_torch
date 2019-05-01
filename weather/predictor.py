import torch
import torch.nn as nn

from weather.dataset import PreWeatherDataset


def create_dict(values, columns):
    result = {}
    for column, value in zip(columns, values):
        result[column] = value
    return result


def weather_predict(net: nn.Module, pre_weather: PreWeatherDataset):
    net.eval()

    with torch.no_grad():
        enc_inputs, dec_inputs = pre_weather[0].values()
        enc_inputs, dec_inputs = enc_inputs.unsqueeze(0).float(), dec_inputs.unsqueeze(0).float()
        outputs, _, _ = net(enc_inputs, dec_inputs)
        pre_weather.append(outputs.squeeze(0).numpy())

    return pre_weather.weather
