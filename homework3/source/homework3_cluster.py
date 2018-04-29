import itertools
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import matplotlib as mpl


def generate_dict(column):

    value = list(set(column))
    return {value[i]: i for i in range(value.__len__())}


if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\CW\Desktop\课程\数据挖掘\作业三\train.csv', low_memory=False)
    columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
               'Embarked']
    data_copy = data.copy()

    for column in columns:
        data_copy[column] = data_copy[column].replace(generate_dict(data_copy[column]))

    y_data = data_copy['Survived'].as_matrix()
    # 统计得知 'PassengerId', 'Name'两个属性 对于每个对象都有不同的值，则该属性对于模型和任务没有任何帮助，故去除
    # 另外虽然'Ticket'属性值不是一一对应乘客，但是实际情况应该是一一对应(满射)的，之所以不是应该是数据统计上的失误，故去除
    # 最终对模型有意义的属性为以下8个
    x_data = data_copy[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].as_matrix()

    # 原始维度数据分类
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.9, test_size=0.1)

    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)

    # k-means 聚类
    km = KMeans(init="k-means++", n_clusters=2, random_state=28)
    km.fit(X_tsne)
    y_km = km.predict(X_tsne)
    km_center = km.cluster_centers_

    # batch k-means 聚类
    mini_km = MiniBatchKMeans(init="k-means++",n_clusters=2,batch_size=10,random_state=28)
    mini_km.fit(X_tsne)
    y_mini_km = mini_km.predict(X_tsne)
    mini_km_center = mini_km.cluster_centers_

    # 密度聚类
    y_db = DBSCAN(eps=0.3, min_samples=2).fit_predict(X_tsne)


    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    # cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])
    cm2 = mpl.colors.ListedColormap(['#00FF00', '#FF0000'])

    plt.title("original scatter graph")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_data, edgecolors='k', cmap=cm_light)
    plt.savefig("原始图" + ".png")
    plt.show()

    plt.title("k-means")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_km, edgecolors='k', cmap=cm_light)
    plt.scatter(km_center[:, 0], km_center[:, 1], c=range(2), s=60, cmap=cm2,
                edgecolors='none')
    plt.savefig("km" + ".png")
    plt.show()

    plt.title("batch k-means")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_mini_km, edgecolors='k', cmap=cm_light)
    plt.scatter(mini_km_center[:, 0], mini_km_center[:, 1], c=range(2), s=60, cmap=cm2,
                edgecolors='none')
    plt.savefig("batch km" + ".png")
    plt.show()

    plt.title("DBSCAN")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=30, c=y_db, edgecolors='k')
    plt.savefig("DBSCAN" + ".png")
    plt.show()

    plt.close()







