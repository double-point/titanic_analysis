# encoding:utf-8
# FileName: tools
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/3/29 20:05
# Description: 部分工具

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def sns_set():
    """
    sns 相关设置
    @return:
    """
    # 声明使用 Seaborn 样式
    sns.set()
    # 有五种seaborn的绘图风格，它们分别是：darkgrid, whitegrid, dark, white, ticks。默认的主题是darkgrid。
    sns.set_style("whitegrid")
    # 有四个预置的环境，按大小从小到大排列分别为：paper, notebook, talk, poster。其中，notebook是默认的。
    sns.set_context('talk')
    # 中文字体设置-黑体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    # 解决Seaborn中文显示问题并调整字体大小
    sns.set(font='SimHei')

    return sns


def get_mode_age(pclass, embarked, sex, data):
    """
    使用同等舱位、同一登船港口、同一性别的众数 / 均值进行填充年龄
    @param pclass:
    @param embarked:
    @param sex:
    @param data:
    @return:
    """
    data = data.loc[(data['Pclass'] == pclass) & (data['Embarked'] == embarked) & (data['Sex'] == sex), ]
    # 使用众数进行填充
    if data.size < 1:
        return '无法填充'
    return data['Age'].mode()[0]


def get_person_tag(age, sex):
    """
    针对年龄和性别对人群分类
    @param age:
    @param sex:
    @return:
    """
    # 儿童/年轻人/中年人 标签
    child_age = 15
    adult_age = 30

    if age < child_age:
        return 'child'
    elif age < adult_age:
        if sex == 'male':
            return 'young_male'
        else:
            return 'young_female'
    else:
        if sex == 'male':
            return 'adult_male'
        else:
            return 'adult_female'


if __name__ == '__main__':
    pass