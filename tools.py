import sys
import datetime
import json
import csv


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


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r' + s)
    sys.stdout.flush()


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')


def json_to_dict(file_path):
    with open(file_path, 'r') as f:
        dict = json.load(fp=f)
    return dict


def people_str_to_kinship(input1, input2):
    """
    根据输入的p1，p2的人名和csv中读取到的对应关系判断是否有kinship
    :param input1: 第一个人的人名
    :param input2: 第二个人的人名
    :return: 两人的关系，若为同一人则返回-1，若不为同一人但有亲属关系则返回1，若不为同一人且无亲属关系则返回0
    """
    if input1 == input2:
        return -1
    else:
        with open("new_label.csv") as f:
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


def people_id_to_name(id, tr_pairs):
    """
    根据人的id和training set，获取人名
    :param id: 人的id
    :param tr_pairs: training set
    :return: 人名，如"F0009/MID2"，若找不到id对应的人名则返回None
    """
    for pair in tr_pairs:
        if pair[1] == id:
            path = pair[0]
            return path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)]
    return None


def people_ids_to_kinship(id1, id2, tr_pairs):
    """
    根据输入的p1，p2的人的id，得到其亲属关系
    该方法是对people_str_to_kinship在使用id作为输入时的包装
    :param id1: p1的id
    :param id2: p2的id
    :param tr_pairs: 训练集
    :return: 亲属关系，若为同一人则返回-1，若不为同一人但有亲属关系则返回1，若不为同一人且无亲属关系则返回0
    """
    name1 = people_id_to_name(id1, tr_pairs)
    name2 = people_id_to_name(id2, tr_pairs)
    kinship = people_str_to_kinship(name1, name2)
    return kinship


def img_paths_to_kinship(path1,path2):
    """
    根据输入的两张图的图片路径，得到其亲属关系
    该方法是对people_str_to_kinship在使用图片路径作为输入时的包装
    :param path1: 第一张图片的path
    :param path2: 第二张图片的path
    :return: 亲属关系，若为同一人则返回-1，若不为同一人但有亲属关系则返回1，若不为同一人且无亲属关系则返回0
    """
    input1 = path1[findStr(path1, "/", 1) + 1:findStr(path1, "/", 3)]
    input2 = path2[findStr(path2, "/", 1) + 1:findStr(path2, "/", 3)]
    kinship = people_str_to_kinship(input1, input2)
    return kinship


