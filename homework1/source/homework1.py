import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import json
import copy

from multiprocessing import Process

# 创建pandas文件读取对象
def factory(file):
    csv_file = pd.read_csv(file, low_memory=False)
    return csv_file

# 标称属性统计
def nominal_statistic(csv_file, numeric_attr, name):
    result_dict = {}
    for column in csv_file.columns:
        if column not in numeric_attr:
            result_dict[column] = csv_file[column].value_counts().to_dict()
    json.dump(result_dict, open(r'result/'+name+'_nominal_attr.json', 'w', encoding='utf-8'))


# 数值属性统计
def numeric_statistic(csv_file, numeric_attr, name):
    result_dict = {}
    for column in numeric_attr:
        column_series = copy.copy(csv_file[column])
        clean_series = column_series.dropna()

        num_of_NaN = column_series.__len__() - clean_series.__len__()

        clean_list = clean_series.values.tolist()

        clean_list.sort()
        len = clean_list.__len__()
        max_value = clean_list[-1]
        min_value = clean_list[0]
        sum_value = sum(clean_list)
        mean_value = sum_value / clean_list.__len__()

        Q1 = clean_list[int((len + 1) * 0.25)]
        Q2 = clean_list[int((len + 1) * 0.5)]
        Q3 = clean_list[int((len + 1) * 0.75)]

        result = [max_value, min_value, mean_value, Q2, [Q1, Q2, Q3], num_of_NaN]
        result_dict[column] = result
    json.dump(result_dict, open('result/'+name+'_numeric_attr.json', 'w', encoding='utf-8'))

# 数据清洗
def clean_data(csv_file, column, percent):
    # 去除缺失值
    values_dropna = csv_file[column].dropna().values
    values_count = csv_file[column].dropna().value_counts()
    values_clean = list(values_dropna)

    # 去除频率为1的值
    # for value, count in values_count.iteritems():
    #     if count == 1:
    #         values_clean.remove(value)

    # 为加快速度，对所有取值种类的频数-1，近似等效于去除频率为1的值
    for item in values_count.index:
        values_clean.remove(item)

    values_clean.sort()
    len = values_clean.__len__()

    # 按percent比例截尾
    vc = values_clean[int(len * percent):int(len * (1 - percent))]

    return values_dropna, values_clean, vc

# 画图
def draw_numeric(csv_file, numeric_attr):
    for column in numeric_attr:
        print("clean_before")
        values_dropna, values_clean, vc = clean_data(csv_file, column, 0.05)
        print("clean_over")
        loc = 'graph/'
        draw_hist(column, vc, loc)
        print("hist_over")
        draw_qq_norm(column, vc, loc)
        print("qq_over")
        draw_box(column, values_clean, loc)

# 盒图
def draw_box(column, values_clean, loc):
    plt.figure(figsize=(2.8,2))
    # 离散点 (图标样式,图标颜色,大小,..)
    fp = {'marker': "o", 'markerfacecolor': 'blue', 'markersize': 5, 'linestyle': 'none'}
    plt.title("Box:" + str(column))
    plt.boxplot(values_clean, flierprops=fp)
    plt.savefig(loc+'box_'+column+'.png')
    # plt.show()
    plt.close()
    pass

# 直方图
def draw_hist(column, vc, loc):
    plt.figure(figsize=(2.8, 2))
    plt.title("Hist:" + str(column))
    plt.hist(vc, bins=20)
    plt.savefig(loc+'hist_'+column+'.png')
    plt.close()
    pass

# qq图检测是否为正态分布
def draw_qq_norm(column, vc, loc):
    plt.figure(figsize=(2.8, 2))
    stats.probplot(vc, dist="norm", plot=plt)
    plt.title("Q-Q:" + str(column))
    plt.savefig(loc+'qq_'+column+'.png')
    # plt.show()
    plt.close()
    pass

# qq图检测两属性间的相关度
def draw_qq_double(csv_file, double_column):
    data = csv_file[list(double_column)].dropna()
    x = data[double_column[0]].values
    y = data[double_column[1]].values

    plt.figure(figsize=(2.8,2))
    plt.title(double_column[0] + "_" + double_column[1])
    plt.plot(x, y, 'ro')
    plt.savefig('graph/comparison/'+double_column[0]+"_"+double_column[1]+'.png')
    plt.show()
    plt.close()


# 去除缺失值 绘图函数
def complete_dropna(csv_file, column):
    loc = "graph/complete/type1_"
    values_dropna = csv_file[column].dropna().values
    draw_hist(column, values_dropna, loc)
    draw_qq_norm(column, values_dropna, loc)
    draw_box(column, values_dropna, loc)
    pass

# 用最高频率值来填补缺失值 绘图函数
def complete_fre_attr(csv_file, column):
    value_count = csv_file[column].dropna().value_counts()
    max_fre_value = value_count.index[0]
    data = csv_file[column]
    miss_index = data[data.isnull()].index
    complete_data = data.copy()
    for i in miss_index:
        complete_data[i] = max_fre_value

    loc = "graph/complete/type2_"
    draw_hist(column, complete_data, loc)
    draw_qq_norm(column, complete_data, loc)
    draw_box(column, complete_data, loc)

# 通过属性的相关关系来填补缺失值 绘图函数
def complete_rel_attr(csv_file, double_column):
    target_data = csv_file[double_column[0]]
    source_data = csv_file[double_column[1]]
    flag1 = target_data.isnull().values
    flag2 = source_data.isnull().values
    complete_data = target_data.copy()
    for index, value in target_data.iteritems():
        if flag1[index] == True and flag2[index] == False:
            # x = y
            # complete_data[index] = source_data[index]
            # x = y-1
            complete_data[index] = 1 - source_data[index]

    values_clean = list(complete_data.dropna().values)

    # 去除频率为1的值
    for value, count in complete_data.value_counts().iteritems():
        if count == 1:
            values_clean.remove(value)

    loc = "graph/complete/type3_"
    draw_hist(double_column[0], values_clean, loc)
    draw_qq_norm(double_column[0], values_clean, loc)
    draw_box(double_column[0], values_clean, loc)

# 查找两个对象间相异度最小的 指定的 column值
def find_dis_value(csv_file, pos, column, numeric_attr):

    def dis_objs(tar_obj, sou_obj):
        dis_value = 0
        count = 0
        for column in tar_obj.index:
            if tar_obj[column] != np.NaN and sou_obj[column] != np.NaN:
                if column in numeric_attr:
                        values_sort = csv_file[column].dropna().values.sort()
                        denominator = values_sort[-1] - values_sort[0]
                        dis_value += abs(tar_obj[column] - sou_obj[column])/denominator
                        count += 1

                elif tar_obj[column] == sou_obj[column]:
                    dis_value += 1
                count += 1
            else:
                continue
        return dis_value/count

    mindis = 9999
    result_pos = -1
    target_obj = csv_file.ix[pos]
    for index in csv_file.index:
        if index == pos:
            continue
        source_obj = csv_file.ix(index)
        tmp = dis_objs(target_obj, source_obj)
        if tmp < mindis:
            result_pos = index
    return result_pos

# 通过数据对象之间的相似性来填补缺失值 绘图函数
def complete_smi_attr(csv_file, column, numeric_attr):
    data = csv_file[column].copy()
    for index, value in data.iteritems():
        if value == np.NaN:
            data[index] = data[find_dis_value(csv_file, index, column, numeric_attr)]
    loc = "graph/complete/type4_"
    draw_hist(column, data.dropna().values, loc)
    draw_qq_norm(column, data.dropna().values, loc)
    draw_box(column, data.dropna().values, loc)

# 选择数值属性
def select_attr(csv_file):
    nominal_attr= []
    columns = csv_file.columns.values
    for column in columns:
        values = csv_file[column].dropna().values
        value_count = csv_file[column].dropna().value_counts()
        if value_count.__len__()<10:
            nominal_attr.append(column)
            continue
        for value in values[:10]:
            try:
                if float(value) > 9999:
                    nominal_attr.append(column)
                    break
            except BaseException as e:
                nominal_attr.append(column)
                break

    return list(set(columns)-set(nominal_attr))


if __name__ == "__main__":
    homework_file1 = r"C:\Users\CW\Desktop\课程\数据挖掘\作业一\Building_Permits.csv"
    homework_file2 = r"C:\Users\CW\Desktop\课程\数据挖掘\作业一\NFL Play by Play 2009-2017 (v4).csv"

    # ----------数据集1---------------
    # 人工选择所有数值属性
    numeric_attr1 = ['Number of Existing Stories', 'Number of Proposed Stories', 'Estimated Cost', 'Revised Cost',
                    'Existing Units', 'Proposed Units']#'Location'
    csv_file1 = factory(homework_file1)

    nominal_statistic(csv_file1, numeric_attr1, "Building_Permits")
    numeric_statistic(csv_file1, numeric_attr1, "Building_Permits")
    draw_numeric(csv_file1, numeric_attr1)

    # 绘制两个属性的qq图，判断相关性
    for double_column in itertools.combinations(numeric_attr1, repeat=2):
        draw_qq_double(csv_file1, double_column)

    # 只展示Estimated Cost属性填补后的效果
    complete_dropna(csv_file1, 'Estimated Cost')
    complete_fre_attr(csv_file1, 'Estimated Cost')
    complete_rel_attr(csv_file1, ['Estimated Cost', 'Revised Cost'])
    complete_smi_attr(csv_file1, 'Estimated Cost', numeric_attr1)

    # 数据集1和2需要各自单独运行代码一遍
    # ----------数据集2---------------
    # csv_file2 = factory(homework_file2)
    # 找出所有数值属性
    # numeric_attr2 = select_attr(csv_file2)
    # 人工选取若干数值属性
    # numeric_attr2 = ['Away_WP_pre', 'FieldGoalDistance', 'Field_Goal_Prob', 'Home_WP_pre', 'Opp_Field_Goal_Prob', 'Touchdown_Prob', 'yrdline100', 'yrdln']
    # nominal_statistic(csv_file2, numeric_attr2, "NFL Play by Play 2009-2017 (v4)")
    # numeric_statistic(csv_file2, numeric_attr2, "NFL Play by Play 2009-2017 (v4)")
    # draw_numeric(csv_file2, numeric_attr2)

    # 为加快速度，使用多进程
    # def child(csv_file, column):
    #     draw_numeric(csv_file, [column])
    #
    #
    # childs = []
    # for column in numeric_attr2:
    #     p = Process(target=child, args=(csv_file2, column))
    #     p.start()
    #     childs.append(p)
    #
    # for child_p in childs:
    #     child_p.join()

    # for double_column in itertools.combinations(numeric_attr2, 2):
    #     draw_qq_double(csv_file2, double_column)

    # 只展示Home_WP_pre属性填补后的效果
    # complete_dropna(csv_file2, 'Home_WP_pre')
    # complete_fre_attr(csv_file2, 'Home_WP_pre')
    # complete_rel_attr(csv_file2, ['Home_WP_pre', 'Away_WP_pre'])
    # complete_smi_attr(csv_file2, 'Away_WP_pre', numeric_attr2)
