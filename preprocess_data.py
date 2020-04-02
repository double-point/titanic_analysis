# encoding:utf-8
# FileName: preprocess_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/3/29 12:51
# Description: 数据预处理
import re

import pandas as pd

# 显示所有列
from read_data import read_data
from tools import get_mode_age

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def process_data(df_data):
    """
    数据预处理
    @param df_data:
    @return:
    """
    """
    其中，各个字段分别是：
    数据特征：
        PassengerId： 乘客ID
        Pclass： 乘客等级(1/2/3等舱位)
        Name： 乘客姓名
        Sex： 性别
        Age： 年龄
        SibSp： 堂兄弟/妹个数
        Parch： 父母与小孩个数
        Ticket： 船票信息
        Fare： 票价
        Cabin： 客舱
        Embarked： 登船港口
    目标信息：
        Survived:  生还
    """

    """
    1. 数据分布（缺失、异常、数值型、离散型）
    """
    print(df_data)
    # 通过.columns查看列名
    print(df_data.columns)

    # 通过info 查看数据整体情况（不区分测试集和训练集）
    print(df_data.info())
    # 也可已通过以下方法直接判断缺失情况
    print(df_data.isnull().sum())

    # 通过describe 查看数值型数据的分布
    print(df_data.describe())

    """
    2. 数据拆分、合并，字段分析
    """
    df_data['last_name'] = df_data['Name'].apply(lambda name: name.split(',')[0])
    df_data['first_name'] = df_data['Name'].apply(lambda name: name.split(',')[1])

    # 通过正则表达式分离出名称前面的title标识
    df_data['Title'] = df_data['first_name'].apply(
        lambda first_name: re.search('([A-Za-z]+)\.', first_name, flags=0).group(1))
    # 查看都有哪些Title
    print(df_data['Title'].value_counts())

    title_mapping = {'Mr': 1, 'Miss': 2, 'Mlle': 2, 'Ms': 2, 'Mrs': 2, 'Mme': 2,
                     'Rev': 3, 'Dr': 3, 'Col': 4, 'Major': 4, 'Capt': 4,
                     'Master': 5, 'Don': 5, 'Dona': 5, 'Lady': 5, 'Countess': 5, 'Jonkheer': 5, 'Sir': 5,
                     }
    # 将Title直接量化
    df_data['TitleType'] = df_data['Title'].map(title_mapping)
    # 新增名字长度列
    df_data['Namelen'] = df_data['Name'].apply(lambda name: len(name))

    # 家庭成员数：+1 表示加上自己
    df_data["Numbers"] = df_data["SibSp"] + df_data["Parch"] + 1

    """
    3. 缺失数据填补
    """
    """Fare缺失填充"""
    # S登船港口3等船舱的 Storey, Mr. Thomas，年龄是60.5岁，家庭团体只有他一个人
    print(df_data.loc[df_data['Fare'].isnull(), :])
    # 选择S登船港口3等船舱年龄是大于60岁的平均票价填充Fare
    df_data.loc[df_data['Fare'].isnull(), 'Fare'] = df_data.loc[(df_data['Embarked'] == 'S') &
                                                                (df_data['Pclass'] == 3) &
                                                                (df_data['Age'] >= 60), 'Fare'].mean()
    """Embarked缺失填充"""
    # 1等船舱的B28客舱的两位女士，一个38岁，一个62岁
    print(df_data.loc[df_data['Embarked'].isnull(), :])
    # 1等船舱的女性登录最多的港口分别为C：71、S：69：Q：2
    print(df_data.loc[(df_data['Pclass'] == 1) & (df_data['Sex'] == 'female'), 'Embarked'].value_counts())
    # 选择1等船舱的女性登录最多的港口C进行填充
    df_data.loc[df_data['Embarked'].isnull(), 'Embarked'] = 'C'

    """Age缺失填充"""
    # 使用同等舱位、同一登船港口、同一性别的众数 / 均值进行填充
    df_data.loc[df_data['Age'].isnull(), 'Age'] = \
        df_data.loc[df_data['Age'].isnull(), ['Pclass', 'Embarked', 'Sex']].apply(
            lambda info: get_mode_age(info[0], info[1], info[2], df_data), axis=1)

    """Cabin缺失填充"""
    # 缺失值过多，无法直接填充，可以直接将缺失值当做一类数据，这里直接置为Null，非空为NotNull
    df_data['CabinType'] = df_data['Cabin'].isnull().apply(lambda x: 'Null' if x is True else 'NotNull')

    return df_data


if __name__ == '__main__':
    pass
