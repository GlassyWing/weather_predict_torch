from glob import glob
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 自相关和偏相关图
def draw_acf_pacf(ts, lags=31):
    f, ax = plt.subplots(2, 1)
    plot_acf(ts, lags=lags, ax=ax[0])
    plot_pacf(ts, lags=lags, ax=ax[1])
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.show()

def test_stationarity(timeseries):
    # 定义滚动统计数据
    rolmean = timeseries.rolling(6).mean()
    rolstd = timeseries.rolling(6).std()
    # 显示滚动数据
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('滚动平均数和方差')
    plt.show(block=False)
    # DF检验
    print('DF检验结果')
    dftest = adfuller(timeseries, autolag='AIC')
    # 打印p值
    print(dftest[1])


if __name__ == '__main__':
    file_paths = glob(f'../data/weather/*.csv')
    df = pd.read_csv('../data/weather.csv')

    df: DataFrame = df["24_hours_avg_tem"]
    df.diff(1)

    draw_acf_pacf(df.diff().dropna(), 30)
    test_stationarity(df[:2000])
