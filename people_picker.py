from tools import findStr
import csv
import random
import os
import pandas as pd
import json


def get_tr_pairs():
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
    return pairs


def people_picker(a_index, tr_pairs):
    # a_index是人的id
    # 获取a的人名
    a_path = ''
    for pair in tr_pairs:
        if pair[1] == a_index:
            a_path = pair[0]
    a_name = a_path[findStr(a_path, "/", 1) + 1:findStr(a_path, "/", 3)]

    # 获取所有人的人名
    people_name = []
    for pair in tr_pairs:
        path = pair[0]
        people_name.append(path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)])
    people_name = list(set(people_name))

    # 读取关系表
    with open("new_label.csv") as f:
        reader = csv.reader(f)
        column1 = []
        column2 = []
        for row in reader:
            column1.append(row[0])
            column2.append(row[1])

    # 找到和a有亲属关系的人的名称
    p_list = []
    for id, name in enumerate(column1):
        if name == a_name:
            p_list.append(column2[id])
    for id, name in enumerate(column2):
        if name == a_name:
            p_list.append(column1[id])

    # 找到和a没有亲属关系的人的名称
    n_list = [x for x in people_name if x not in p_list]

    # 随机抽取p，n1，n2的名称
    p_name = random.choice(p_list)
    n1_name = random.choice(n_list)
    n2_name = random.choice(n_list)
    n3_name = random.choice(n_list)
    # print("a: ", a_name)
    # print("p: ", p_name)
    # print("n: ", n1_name)
    # print("n: ", n2_name)

    return a_name, p_name, n1_name, n2_name, n3_name


# 根据人名和trainning set返回对应的人的id
def name_to_id(name, tr_pairs):
    for pair in tr_pairs:
        pair_name = pair[0][findStr(pair[0], "/", 1) + 1:findStr(pair[0], "/", 3)]
        if name == pair_name:
            return pair[1]
    return -1


# 整合上面的两个方法，返回apnn的人的id
def get_apnn_from_id(id, tr_pairs):
    a_name, p_name, n1_name, n2_namei, n3_name = people_picker(id, tr_pairs)
    pids = []
    pids.append(name_to_id(a_name, tr_pairs))
    pids.append(name_to_id(p_name, tr_pairs))
    pids.append(name_to_id(n1_name, tr_pairs))
    pids.append(name_to_id(n2_name, tr_pairs))
    pids.append(name_to_id(n3_name, tr_pairs))

    return pids

# 根据a的id和已经读取的apnn关系的list， 返回对应的apnn的id
def get_apnn_from_id_with_dict(a_id, tr_pairs, all_pn_dict):
    # 由id获取名字
    for pair in tr_pairs:
        pair_id = pair[1]
        if pair_id == a_id:
            path = pair[0]
            name = path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)]
            break

    # 从all_pn_dict中取出当前a的p和n list
    p_list = all_pn_dict[name]['p_list']
    n_list = all_pn_dict[name]['n_list']

    # 从p和n的list中抽取一个p，两个n，并在training set中找到其对应的id
    # 如果抽到的p或n不存在于training set，则重新抽取这个p或n
    p_id = -1
    n1_id = -1
    n2_id = -1
    n3_id = -1
    while p_id == -1:
        p_name = random.choice(p_list)
        p_id = name_to_id(p_name, tr_pairs)
    while n1_id == -1:
        n1_name = random.choice(n_list)
        n1_id = name_to_id(n1_name, tr_pairs)
    while n2_id == -1:
        n2_name = random.choice(n_list)
        n2_id = name_to_id(n2_name, tr_pairs)
    while n3_id == -1:
        n3_name = random.choice(n_list)
        n3_id = name_to_id(n3_name, tr_pairs)


    return [a_id, p_id, n1_id, n2_id, n3_id]


def table_clean(tr_pairs):
    # 获取所有人的人名
    people_name = []
    for pair in tr_pairs:
        path = pair[0]
        people_name.append(path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)])
    people_name = list(set(people_name))

    # 读取关系表
    with open("train_relationships.csv") as f:
        reader = csv.reader(f)
        column1 = []
        column2 = []
        for row in reader:
            column1.append(row[0])
            column2.append(row[1])

    column1_new1 = []
    column2_new1 = []
    for id, name in enumerate(column1):
        if name in people_name:
            column1_new1.append(column1[id])
            column2_new1.append(column2[id])

    column1_new2 = []
    column2_new2 = []
    for id, name in enumerate(column2_new1):
        if name in people_name:
            column1_new2.append(column1_new1[id])
            column2_new2.append(column2_new1[id])

    table = []
    for i in range(column1_new2.__len__()):
        table.append([column1_new2[i], column2_new2[i]])

    list_to_csv(table, "new_label.csv")


# 把list存入csv
def list_to_csv(data, filename):
    df = pd.DataFrame(data=data)
    df.to_csv(filename, encoding="utf8", header=None, index=None)


def data_clean(tr_pairs):
    # 获取所有人的人名
    people_name = []
    for pair in tr_pairs:
        path = pair[0]
        people_name.append(path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)])
    people_name = list(set(people_name))

    # 读取关系表
    with open("new_label.csv") as f:
        reader = csv.reader(f)
        column1 = []
        column2 = []
        for row in reader:
            column1.append(row[0])
            column2.append(row[1])

    for name in people_name:
        if name not in column1 and name not in column2:
            print(name)


# TODO: 所有人pn存为json
def all_pn_json():
    tr_pairs = get_tr_pairs()
    all_pn_dict = {}

    # 获取所有人的人名
    people_name = []
    for pair in tr_pairs:
        path = pair[0]
        people_name.append(path[findStr(path, "/", 1) + 1:findStr(path, "/", 3)])
    people_name = list(set(people_name))

    print("total number: ", people_name.__len__())

    # 读取关系表
    with open("new_label.csv") as f:
        reader = csv.reader(f)
        column1 = []
        column2 = []
        for row in reader:
            column1.append(row[0])
            column2.append(row[1])

    i = 1
    for name in people_name:
        print("current: ", i)
        name_dict = {}

        # 找到有亲属关系的人的名称
        p_list = []
        for id, table_name in enumerate(column1):
            if table_name == name:
                p_list.append(column2[id])
        for id, table_name in enumerate(column2):
            if table_name == name:
                p_list.append(column1[id])

        # 找到没有亲属关系的人的名称
        n_list = [x for x in people_name if x not in p_list]

        name_dict['p_list'] = p_list
        name_dict['n_list'] = n_list

        all_pn_dict[name] = name_dict

        i += 1

    # 存为json
    dict_to_json("all_pn_json.json", all_pn_dict)


def dict_to_json(file_path, dict):
    with open(file_path, "w") as f:
        json.dump(dict, f)


def json_to_dict(file_path):
    with open(file_path, 'r') as f:
        dict = json.load(fp=f)
    return dict

def _pluck(pairs, indices):
    ret = []
    for index, pid in enumerate(indices):
        for pair in pairs:
            if pair[1] == pid:
                filepath = pair[0]
                ret.append([filepath, index])
    return ret






if __name__ == '__main__':
    tr_pairs = get_tr_pairs()
    print(tr_pairs)

