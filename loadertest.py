import os
import csv
import torch
import pandas
import time
from sphereface_rfiw_baseline.tools import printoneline, dt


def loader():
    # 把所有人全部读取到pairs中
    label = -1
    filepath = ""
    pairs = []
    for root, dirs, files in os.walk(r"data/"):
        for file in files:
            # 文件所属目录root
            # 文件名file
            if root == filepath:
                pairs.append([str(os.path.join(root, file)).replace("\\", "/"), label])
            else:
                filepath = root
                label += 1
                pairs.append([os.path.join(root, file).replace("\\", "/"), label])

    # 根据指定编号抽取 validation set 所需的pair
    # 剩下的作为 trainning set
    val_indices = [0, 2, 4, 6, 8]
    tr_indices = [x for x in list(range(pairs.__len__())) if x not in val_indices]
    val_pairs = _pluck(pairs, val_indices)
    tr_pairs = _pluck(pairs, tr_indices)

    # region 根据csv文件将抽取的validation pair生成为带label的validation set
    with open("train_relationships.csv") as f:
        reader = csv.reader(f)
        column1 = []
        column2 = []
        for row in reader:
            column1.append(row[0])
            column2.append(row[1])

    val_sample = []

    for pair in val_pairs:
        path1 = pair[0]
        input1 = path1[findStr(path1, "/", 1) + 1:findStr(path1, "/", 3)]
        for pair in val_pairs:
            path2 = pair[0]
            if path1 != path2:
                input2 = path2[findStr(path2, "/", 1) + 1:findStr(path2, "/", 3)]
                label = val_label(input1, input2, column1, column2)
                val_sample.append([path1, path2, label])

    # 去重
    for sample1 in val_sample:
        for sample2 in val_sample:
            if sample1 != sample2 and set(sample1) == set(sample2):
                val_sample.remove(sample1)
    # endregion

    print(pairs)
    print(val_pairs)
    print(tr_pairs)
    print(val_sample)


# 生成[person1, person2, kinship]的关系表
def people_loader():
    # 把所有人全部读取到pairs中
    people = []
    for root, dirs, files in os.walk(r"data/"):
        for file in files:
            filepath = root.replace("\\", "/")
            person = filepath[5:]
            people.append(person)
            break

    people_kinship = []
    print("number of people = ", people.__len__())
    i = 1
    fileno = 1
    for person1 in people:
        printoneline(dt(), 'person no = %d' % i)
        for person2 in people:
            kinship = people_str_to_kinship(person1, person2) or people_str_to_kinship(person2, person1)
            people_kinship.append([person1, person2, kinship])
            if people_kinship.__len__() / 100 == 1:
                filename = 'people_kinship/people_kinship_' + str(fileno) + '.csv'
                list_to_csv(people_kinship, filename)
                fileno += 1
                people_kinship.clear()
        i += 1

    filename = 'people_kinship/people_kinship_' + str(fileno) + '.csv'
    list_to_csv(people_kinship, filename)


def _pluck(pairs, indices):
    ret = []
    for index, pid in enumerate(indices):
        for pair in pairs:
            if pair[1] == pid:
                filepath = pair[0]
                ret.append([filepath, index])
    return ret


# 根据输入的p1，p2和csv中读取到的对应关系返回label
def val_label(input1, input2, column1, column2):
    input1_index1 = [idx for idx, i in enumerate(column1) if i == input1]
    if input1_index1.__len__() != 0:
        for index in input1_index1:
            if column2[index] == input2:
                return 1
    input1_index2 = [idx for idx, i in enumerate(column2) if i == input1]
    if input1_index2.__len__() != 0:
        for index in input1_index2:
            if column1[index] == input2:
                return 1
    return 0


# 判断string中subStr第findCnt次出现的位置，用于从绝对路径中截取符合train_relation的部分
def findStr(string, subStr, findCnt):
    listStr = string.split(subStr, findCnt)
    if len(listStr) <= findCnt:
        return -1
    return len(string) - len(listStr[-1]) - len(subStr)


# 根据输入的p1，p2的字符串和csv中读取到的对应关系判断是否有kinship
def people_str_to_kinship(input1, input2):
    if input1 == input2:
        return -1
    else:
        with open("train_relationships.csv") as f:
            reader = csv.reader(f)
            column1 = []
            column2 = []
            for row in reader:
                column1.append(row[0])
                column2.append(row[1])

        input1_index1 = [idx for idx, i in enumerate(column1) if i == input1]
        if input1_index1.__len__() != 0:
            for index in input1_index1:
                if column2[index] == input2:
                    return 1
        input1_index2 = [idx for idx, i in enumerate(column2) if i == input1]
        if input1_index2.__len__() != 0:
            for index in input1_index2:
                if column1[index] == input2:
                    return 1
    return 0


# 根据training set得到的pairs，输入人物id得到对应的字符串
def people_id_to_name(id, pairs):
    for pair in pairs:
        if pair[1] == id:
            path = pair[0]
            name = path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)]
            return name


# 把list存入csv
def list_to_csv(data, filename):
    df = pandas.DataFrame(data=data)
    df.to_csv(filename, encoding="utf8", header=None, index=None)


def apnn():
    pass


if __name__ == '__main__':
    while True:
        print(1)
        time.sleep(1)
