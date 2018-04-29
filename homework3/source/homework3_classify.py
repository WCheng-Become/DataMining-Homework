import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm,tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pydotplus
from matplotlib import pylab as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split

import os
os.environ["PATH"] += os.pathsep + 'C:/ProgramFiles/Graphviz2.38/bin/'

def generate_dict(column):

    value = list(set(column))
    return {value[i]: i for i in range(value.__len__())}

def decision_tree(x_train, x_test, y_train, y_test, create_graph):

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(x_train, y_train)

    # y = list(y_test)
    # print(clf.predict(x_test.ix[5], y[5]))
    print("决策树准确率:" + str(clf.score(x_test, y_test)))

    if create_graph == True:
        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("result.pdf")
        graph.write_png("tree.png")
    return clf

def my_svm(x_train, x_test, y_train, y_test):

    clf = svm.SVC(C=0.5, kernel='rbf', gamma=0.5)
    clf.fit(x_train, y_train)
    print("svm准确率:" + str(clf.score(x_test, y_test)))
    return clf

def naive_bayes(x_train, x_test, y_train, y_test):

    # 多项式假设
    # clf = MultinomialNB()
    # 高斯假设
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print("朴素贝叶斯准确率:" + str(clf.score(x_test, y_test)))

    return clf

def drew_result(clf, x_data, y_data, title):
    x1_min, x1_max = x_data[:, 0].min()-1, x_data[:, 0].max()+1
    x2_min, x2_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    t1 = np.linspace(x1_min, x1_max, 100)
    t2 = np.linspace(x2_min, x2_max, 100)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)

    y_hat = clf.predict(x_show)  # 预测
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])

    plt.figure(facecolor='w')
    plt.title(title)
    # 预测值的显示
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
    # 等高线图
    # plt.contourf(xx, yy, Z, alpha=0.4)
    # 样本的显示
    plt.scatter(x_data[:, 0], x_data[:, 1], s=30, c=y_data, edgecolors='k', cmap=cm_dark)
    plt.savefig(title+".png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    data = pd.read_csv(r'C:\Users\CW\Desktop\课程\数据挖掘\作业三\train.csv', low_memory=False)
    columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
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

    decision_tree(x_train, x_test, y_train, y_test, True)
    my_svm(x_train, x_test, y_train, y_test)
    naive_bayes(x_train, x_test, y_train, y_test)

    # 降维数据分类
    # 降维后的散点图,不同的降维方式
    X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)
    # X_lda = LinearDiscriminantAnalysis(n_components=2).fit(x_data, y_data).transform(x_data)
    # X_pca = PCA().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(X_tsne, y_data, random_state=1, train_size=0.9, test_size=0.1)
    model1 = decision_tree(x_train, x_test, y_train, y_test, False)
    drew_result(model1, X_tsne, y_data, "2-D decision tree")
    model2 = my_svm(x_train, x_test, y_train, y_test)
    drew_result(model2, X_tsne, y_data, "2-D svm")
    model3 = naive_bayes(x_train, x_test, y_train, y_test)
    drew_result(model3, X_tsne, y_data, "2-D naive bayes")



