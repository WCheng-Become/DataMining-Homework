# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

csv_file = pd.read_csv(r"C:\Users\CW\Desktop\课程\数据挖掘\作业一\Building_Permits.csv", low_memory=False)

# 筛选的属性
column_usable = []
for column in csv_file.columns:
    # 属性的可能取值要在1到10之间
    if 1 < csv_file[column].value_counts().__len__() < 10:
        column_usable.append(column)

# column_usable = ['Permit Type', 'Permit Type Definition', 'Plansets', 'TIDF Compliance', 'Existing Construction Type', 'Existing Construction Type Description', 'Proposed Construction Type', 'Proposed Construction Type Description']
# 简化属性名称
change_column = []

for column in column_usable:
    words = column.split(' ')
    name = ''
    for word in words:
        name += word[0]
    change_column.append(name)

# print(change_column)

# 截取只包含合适属性的数据集
data = csv_file[column_usable]
# 存储最终目标
trans_dict = {}
record_num = data.index.__len__()
for column in column_usable:
    new_line = [""]*record_num
    for index in data.index:
        item = data[column][index]
        try:
            if np.isnan(item):
                new_line[index] = ""
            else:
                # 拼接属性和属性值作为新属性值
                new_line[index] = change_column[column_usable.index(column)] + "_"+ str(item)
        except BaseException as e:
            new_line[index] = change_column[column_usable.index(column)] + "_" + str(item)
    trans_dict[column] = new_line

csv_write = pd.DataFrame(trans_dict)
# print(csv_write)
csv_write.to_csv('association_mining.csv', index=False, header=False)