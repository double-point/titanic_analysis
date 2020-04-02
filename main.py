# encoding:utf-8
# FileName: main
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/3/29 12:52
# Description:main函数
import re

import pandas as pd
import numpy as np

# 显示所有列
from model_tree import model_data
from preprocess_data import process_data
from read_data import read_data
from view_data import view_data

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    filepath_train_data = 'train.csv'
    filepath_test_data = 'test.csv'
    # 读取训练数据和测试数据
    df_train_data = read_data(filepath_train_data)
    df_test_data = read_data(filepath_test_data)
    # 合并训练集和测试集
    df_data = df_train_data.append(df_test_data, sort=False)
    # df_data.reset_index(inplace=True)
    # 数据预处理
    df_data = process_data(df_data)
    print("=" * 50)
    # 数据可视化探索
    df_train_data = df_data[df_data['Survived'].notnull()]
    df_test_data = df_data[df_data['Survived'].isnull()]
    view_data(df_data, df_train_data)
    # 模型训练并输出预测结果
    clf = model_data(df_data)