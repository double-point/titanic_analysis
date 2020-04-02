# encoding:utf-8
# FileName: view_data
# Author:   xiaoyi | 小一
# email:    1010490079@qq.com
# Date:     2020/3/29 12:52
# Description: 数据可视化探索
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 显示所有列
from sklearn import preprocessing
from sklearn.feature_selection import f_classif, SelectKBest

from preprocess_data import process_data
from read_data import read_data
from tools import sns_set, get_person_tag

pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


def view_data(df_data, df_train_data):
    """
    数据可视化探索，主要针对训练数据进行可视化探索
    @param df_data:
    @param df_train_data:
    @return:
    """
    sns = sns_set()
    print(df_data.info())
    """
    确定定性、定量数据
        定性数据包括：定类数据和定序数据，主要包括以下字段：
            PassengerId：乘客id
            Survived：生还or遇难
            Pclass：船舱等级
            Name：乘客姓名
            Sex：性别
            SibSp：堂兄弟/妹个数
            Parch：父母与小孩个数
            Ticket：船票信息
            Cabin：客舱
            Embarked：登船港口
            CabinType：客舱的分类（新增）
            last_name & first_name & Title(新增姓名衍生列)
            Title：姓名的title标识（新增）
            Title_type：Title的分类（新增）
            Namelen：姓名的长度（新增）
            Namelen_type：姓名的长度的分类（新增）
            Numbers：家庭成员数（新增）
            Numbers_type：家庭成员数的分类（新增）
            title_rank：姓名的title标识分类（新增）
        定量数据包括：定距数据和定比数据，主要包括以下字段：
            Age：年龄
            Fare：票价
    
    分析：    
        Pclass、Sex、SibSp、Parch、Embarked类别较少，可以进行条形图可视化
        Age、Ticket为离散数据，可以用折线图、散点图可视化
    """
    """1.1 离散型单特征的分布"""
    fig, axs = plt.subplots(nrows=2, ncols=4)
    sns.countplot(x='Pclass', hue='Survived', data=df_train_data, ax=axs[0, 0])
    sns.countplot(x='Sex', hue='Survived', data=df_train_data, ax=axs[0, 1])
    sns.countplot(x='Title_type', hue='Survived', data=df_train_data, ax=axs[0, 2])
    sns.countplot(x='Embarked', hue='Survived', data=df_train_data, ax=axs[0, 3])
    sns.countplot(x='SibSp', hue='Survived', data=df_train_data, ax=axs[1, 0])
    sns.countplot(x='Parch', hue='Survived', data=df_train_data, ax=axs[1, 1])
    sns.countplot(x='Numbers', hue='Survived', data=df_train_data, ax=axs[1, 2])
    sns.countplot(x='CabinType', hue='Survived', data=df_train_data, ax=axs[1, 3])

    # 添加标题及相关
    axs[0, 0].set_title('各船舱等级的生还情况', fontsize=13)
    axs[0, 1].set_title('各性别的生还情况', fontsize=13)
    axs[0, 2].set_title('各地位的生还情况', fontsize=13)
    axs[0, 3].set_title('各登录港口的生还情况', fontsize=13)
    axs[1, 0].set_title('各堂兄/妹个数的生还情况', fontsize=13)
    axs[1, 1].set_title('各父母与小孩个数的生还情况', fontsize=13)
    axs[1, 2].set_title('各亲人数量的生还情况', fontsize=13)
    axs[1, 3].set_title('客舱缺失与否的生还情况', fontsize=13)
    axs[0, 0].set_xlabel('', fontsize=12)
    axs[0, 1].set_xlabel('', fontsize=12)
    axs[0, 2].set_xlabel('', fontsize=12)
    axs[0, 2].set_xlabel('', fontsize=12)
    axs[0, 3].set_xlabel('', fontsize=12)
    axs[1, 0].set_xlabel('', fontsize=12)
    axs[1, 1].set_xlabel('', fontsize=12)
    axs[1, 2].set_xlabel('', fontsize=12)
    axs[1, 3].set_xlabel('', fontsize=12)
    axs[0, 0].set_ylabel('人数', fontsize=12)
    axs[0, 1].set_ylabel('', fontsize=12)
    axs[0, 2].set_ylabel('', fontsize=12)
    axs[0, 3].set_ylabel('', fontsize=12)
    axs[1, 0].set_ylabel('人数', fontsize=12)
    axs[1, 1].set_ylabel('', fontsize=12)
    axs[1, 2].set_ylabel('', fontsize=12)
    axs[1, 3].set_ylabel('', fontsize=12)
    fig.suptitle('离散型特征下生还情况分布    『by:小一』', fontsize=16)
    plt.show()

    """
    1.1 总结与分析
    总结
        1. 船舱等级为1的生还率最高（生还率0.7），2次之（生还率1/2），3最低（生还率不足1/3）、但3的人数最多
        2. 性别为女性的生还率高（超过2/3）于男性生还率低（不足1/4），男性人数大于女性
        3. 地位为2、3、4、7的生还率高于死亡率，2345分别对应：未婚或没有子女的女士、已婚女士、少爷、有地位者
        4. 登录港口 为C的生还率最高（超过1/2），Q次之（不足1/2），S最低（1/3），但S的人数最多
        5. 堂兄弟/妹个数为1的生还率最高（超过1/2），为0的样本最多，生还率低（1/3）
        6. 父母和孩子个数为1和2的生还率最高（超过1/2），为0的样本最多，生还率低（1/3）
        7. 亲人数量为2、3、4的生还率高于死亡率，只有自已一个人的团体最多，生还率低（不足1/3）
        8. 客舱非空的生还率高于客舱为空的
    分析
        1. 船舱等级为3的人数最多，生还率最低，推测为普通人较多，为普通阶层
        2. 性别对生还率影响较大，女性生还率大于男性
        3. 登录港口，亲人数量对生还率有影响
    """

    """1.2 数值型单特征的分布"""
    fig, axs = plt.subplots(nrows=1, ncols=2)
    # Age的kde分布
    sns.kdeplot(df_train_data.loc[(df_train_data['Survived'] == 0), 'Age'], color='gray', shade=True,
                label='not survived', ax=axs[0])
    sns.kdeplot(df_train_data.loc[(df_train_data['Survived'] == 1), 'Age'], color='g', shade=True,
                label='survived', ax=axs[0])
    # Fare的kde分布
    sns.kdeplot(df_train_data.loc[(df_train_data['Survived'] == 0), 'Fare'], color='gray', shade=True,
                label='not survived', ax=axs[1])
    sns.kdeplot(df_train_data.loc[(df_train_data['Survived'] == 1), 'Fare'], color='g', shade=True,
                label='survived', ax=axs[1])
    # 添加标题及相关
    axs[0].set_title('Age特征的分布', fontsize=13)
    axs[1].set_title('Fare特征的分布', fontsize=13)
    axs[0].set_xlabel('Age', fontsize=12)
    axs[1].set_xlabel('Fare', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)
    axs[1].set_ylabel('', fontsize=12)
    fig.suptitle('数值型特征的生还情况分布    『by:小一』', fontsize=16)
    plt.show()
    """
    1.2 总结
    总结
        1. 通过离中趋势得出中年人占据多数，年龄段在15-47
        2. 通过存活率得出儿童和老人存活率高，儿童和中年人的年龄段分割点是15岁
        3. 通过离中趋势得出票价在0-100较多
        4. 通过存活率得出30是分界点，票价高于30，生还率大于死亡率
    """

    """2.1 多特征的分布"""
    # 箱型图特征分析
    fig, axs = plt.subplots(nrows=2, ncols=2)
    # 不同船舱等级下年龄的分布
    sns.boxenplot(x='Pclass', y='Age', hue='Sex', data=df_train_data, ax=axs[0, 0])
    # 不同登录港口下年龄的分布
    sns.boxenplot(x='Embarked', y='Age', hue='Sex', data=df_train_data, ax=axs[0, 1])
    # 不同船舱等级下票价的分布
    sns.boxenplot(x='Pclass', y='Fare', hue='Sex', data=df_train_data, ax=axs[1, 0])
    # 不同登录港口下票价的分布
    sns.boxenplot(x='Embarked', y='Fare', hue='Sex', data=df_train_data, ax=axs[1, 1])
    # 添加标题及相关
    axs[0, 0].set_title('不同船舱等级下年龄的分布图', fontsize=13)
    axs[0, 1].set_title('不同登陆港口下年龄的分布图', fontsize=13)
    axs[1, 0].set_title('不同船舱等级下票价的分布图', fontsize=13)
    axs[1, 1].set_title('不同登陆港口下票价的分布图', fontsize=13)
    axs[0, 0].set_xlabel('')
    axs[0, 1].set_xlabel('')
    axs[0, 0].set_ylabel('年龄', fontsize=12)
    axs[1, 0].set_ylabel('票价', fontsize=12)
    axs[0, 1].set_ylabel('')
    axs[1, 1].set_ylabel('')

    fig.suptitle('年龄与票价特征的分布情况     『by:小一』', fontsize=16)
    plt.show()

    # Sex，Pclass分类条件下的 Age年龄对Survived的散点图
    grid = sns.FacetGrid(df_train_data, col='Pclass', row='Sex', hue='Survived', palette='seismic')
    grid = grid.map(plt.scatter, 'PassengerId', 'Age')
    grid.add_legend()
    plt.show()

    # Sex，Embarked分类条件下的 Age年龄对Survived的散点图
    grid = sns.FacetGrid(df_train_data, col='Embarked', row='Sex', hue='Survived', palette='seismic')
    grid = grid.map(plt.scatter, 'PassengerId', 'Age')
    grid.add_legend()
    plt.show()

    # Embarked，Pclass分类条件下的 Age年龄对Survived的散点图
    grid = sns.FacetGrid(df_train_data, col='Embarked', row='Pclass', hue='Survived', palette='seismic')
    grid = grid.map(plt.scatter, 'PassengerId', 'Age')
    grid.add_legend()
    plt.show()

    df_train_data = df_train_data.loc[df_train_data['Fare'] < 300, :]
    # Embarked，Pclass分类条件下的 Fare票价对Survived的散点图
    grid = sns.FacetGrid(df_train_data, col='Embarked', row='Pclass', hue='Survived', palette='seismic')
    grid = grid.map(plt.scatter, 'PassengerId', 'Fare')
    grid.add_legend()
    plt.show()

    """通过画图进行特征选择"""
    sns.heatmap(df_data.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=sns.color_palette('RdBu', n_colors=256),
                linecolor='white', annot=True)
    plt.title('the feature of corr')
    plt.show()

    # 对年龄字段进行处理
    df_train_data['Age'] = df_train_data.loc[:, ['Age', 'Sex']].apply(lambda x: get_person_tag(x[0], x[1]), axis=1)
    # 对Number 字段进行处理
    df_train_data['Numbers'] = pd.cut(df_train_data['Numbers'], bins=[0, 1, 2, 3, 20], labels=[0, 1, 2, 3])

    """选择最优的特征"""
    # 将类别特征转换为量化特征
    select_features = ['Sex', 'Embarked', 'CabinType', 'Age']
    for feature in select_features:
        le = preprocessing.LabelEncoder()
        le.fit(df_train_data[feature])
        df_train_data[feature] = le.transform(df_train_data[feature])
    # 确定特征变量和标签变量
    features = df_train_data.columns.values.tolist()
    label = df_train_data["Survived"].values
    # 删除类别型特征，保留重要特征
    for feature in ['Survived', 'PassengerId', 'Name', 'last_name', 'first_name', 'Embarked', 'Ticket', 'Cabin', 'Title']:
        features.remove(feature)
    df_train_data = df_train_data[features]

    # 选择最优特征
    selector = SelectKBest(f_classif, k=len(features))
    selector.fit(df_train_data, label)
    scores = -np.log10(selector.pvalues_)
    indices = np.argsort(scores)[::-1]
    print("Features importance :")
    for feature in range(len(scores)):
        print("%0.1f %s" % (scores[indices[feature]], features[indices[feature]]))
    # """选中得分score大于2.5的特征"""
    # features_selected = []
    # for feature in range(len(scores)):
    #     if scores[indices[feature]] >= 2.5:
    #         features_selected.append(features[indices[feature]])


if __name__ == '__main__':
    pass