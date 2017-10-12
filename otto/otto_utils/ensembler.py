import csv
import numpy as np
import os
import blender
import consts
import utils


if __name__ == '__main__':
    weights = blender.get_weights()
    prediction_files = utils.get_prediction_files()

    with open(os.path.join(consts.OUTPUT_PATH, 'ensembler_weighted_models.csv'), 'wb') as f_out:  # 输出basemodel预测概率加权后的预测概率值
        writer = csv.writer(f_out)
        readers = []
        f_ins = []
        for fpred in prediction_files:
            f_in = open(os.path.join(consts.ENSEMBLE_PATH, fpred), 'rb')
            f_ins.append(f_in)  # open的文件对象
            readers.append(csv.reader(f_in))  # reader的文件对象
        # Copy header
        writer.writerow(readers[0].next())
        for r in readers[1:]:
            r.next()  # 就是整个文件数据，为什么要用next（）暂时不知道
        # Merge content
        for line in readers[0]:
            file_name = line[0]
            preds = weights[0] * np.array(map(float, line[1:]))  # 首先把第一个文件数据加权后导入，line[1:]就是除了第一行的所有数据
            for i, r in enumerate(readers[1:]):
                preds += weights[i+1] * np.array(map(float, r.next()[1:]))  #r.next()[1:]就是除了第一行的所有数据
            preds /= np.sum(weights)
            writer.writerow([file_name] + list(preds))
        # Close files
        for f_in in f_ins:
            f_in.close()
