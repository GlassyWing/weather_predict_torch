import pandas as pd

if __name__ == '__main__':
    weather = pd.read_csv("../data/weather.csv")
    weather = weather[(weather['year'] == 2017) & (weather['month'] == 5) & (weather['county'] == '仁和')][::-1]

    weather.to_csv("../data/forecast_test.csv", index=False)
