from sphereface_rfiw_baseline.tools import _pluck, findStr, val_label
import os
import csv
import random


def loader():
    # 把所有50个人全部读取到pairs中
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

    print("All:")
    print(pairs)


    # 根据指定人的ID抽取 validation set 所需的pair
    # 剩下的作为 trainning set
    # 此处为抽取最后10个家庭，共56个人
    val_indices = list(range(pairs[-1][1] + 1 - 56, pairs[-1][1] + 1))

    print("val_indices:")
    print(val_indices)

    tr_indices = [x for x in list(range(pairs[-1][1] + 1)) if x not in val_indices]
    val_pairs = _pluck(pairs, val_indices)
    tr_pairs = _pluck(pairs, tr_indices)

    print("val_pairs:")
    print(val_pairs)

    # region 根据csv文件将抽取的validation pair生成为带label的validation set
    with open("new_label.csv") as f:
        reader = csv.reader(f)
        column1 = []
        column2 = []
        for row in reader:
            column1.append(row[0])
            column2.append(row[1])

    val_sample = []

    val_pairs_compare1 = []
    for pair in val_pairs:
        path1 = pair[0]
        input1 = path1[findStr(path1, "/", 1) + 1:findStr(path1, "/", 3)]
        val_pairs_compare1.append(pair)
        val_pairs_compare2 = [x for x in val_pairs if x not in val_pairs_compare1]
        for pair in val_pairs_compare2:
            path2 = pair[0]
            input2 = path2[findStr(path2, "/", 1) + 1:findStr(path2, "/", 3)]
            if input1 != input2:
                label = val_label(input1, input2, column1, column2)
                val_sample.append([path1, path2, label])

    # 去重(上述代码已自动去重)
    # for sample1 in val_sample:
    #     for sample2 in val_sample:
    #         if sample1 != sample2 and set(sample1) == set(sample2):
    #             val_sample.remove(sample1)

    # 让validation set中0和1的数量相同
    number_of_1 = 0
    number_of_0 = 0
    for sample in val_sample:
        if sample[2] == 0:
            number_of_0 += 1
        else:
            number_of_1 += 1
    number_of_0_to_remove = number_of_0 - number_of_1

    print("number_of_0 = ", number_of_0)
    print("number_of_1 = ", number_of_1)
    print("number_of_0_to_remove = ", number_of_0_to_remove)

    number_of_0_removed = 0
    while number_of_0_removed < number_of_0_to_remove:
        sample = random.choice(val_sample)
        if sample[2] == 0:
            val_sample.remove(sample)
            number_of_0_removed += 1
    # endregion
    print(val_sample)


if __name__ == '__main__':
    loader()
