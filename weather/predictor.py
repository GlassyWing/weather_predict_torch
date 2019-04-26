from datetime import timedelta

import torch
import torch.nn as nn
import pandas as pd

from weather.dataset import PreWeatherDataset


def create_dict(values, columns):
    result = {}
    for column, value in zip(columns, values):
        result[column] = value
    return result


def weather_predict(net: nn.Module, pre_weather: PreWeatherDataset, num_future=120):
    net.eval()
    curr_date = pre_weather.get_curr_date()
    place = pre_weather.get_place()
    net.float()
    with torch.no_grad():
        for i in range(num_future):
            last_idx = len(pre_weather) - 1
            input, addition = pre_weather[last_idx].values()
            input, addition = input.unsqueeze(0).float(), addition.unsqueeze(0).float()
            output, _ = net(input, addition)
            output = output[0, -1]

            # 时间累积一天
            curr_date = curr_date + timedelta(days=1)

            output = create_dict(output.numpy(), pre_weather.conclusion)
            output['place'] = place
            output['year'] = curr_date.year
            output['month'] = curr_date.month
            output['day'] = curr_date.day

            if pre_weather.transform is not None:
                output = pd.DataFrame(data=output, index=[0])
                output[pre_weather.according] = pre_weather.transform.I()(output[pre_weather.according])
                output = output.to_dict('index')[0]

            pre_weather.append(output)

    return pre_weather.weather
