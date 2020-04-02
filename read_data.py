# encoding:utf-8
# FileName: read_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/3/29 12:49
# Description: 读取文件

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def read_data(filepath):
    df_data = pd.read_csv(filepath, sep=',')

    return df_data


if __name__ == '__main__':
    filepath_train_data = 'train.csv'
    df_data = read_data(filepath_train_data)
