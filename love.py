import numpy as np
import time
import xlrd
import math
import matplotlib.pyplot as plt
import pandas as pd

'''
    read data from xls file, use avg price as feature (the 5 col in sheet)
'''
def read_data(filepath, col):
    workbook = xlrd.open_workbook(filepath)
    sheet = workbook.sheet_by_index(0)
    data = sheet.col_values(col)
    return data

'''
    Fuzzy Search Predictor
'''
class FuzzyPredictor(object):
    def __init__(self, data_list, labels = 9, batch_size=125, channel=5):
        self.batch_size = batch_size
        self.channel = channel
        self.days = channel
        self.labels = labels
        self.data = data_list
        
        # Model
        # 映射关系
        self.mapping = [{}] * (self.channel + 1)
        # 类别范围, 第0维为min, 第1维为max
        self.range = np.zeros((self.channel + 1, self.days, 2), dtype=float)

    '''
        将一组数据转换为带有时序的矩阵，如[]
        input:  data_list shape:[C, B], e.g. [5, 125]
                channel: 标志输入的data_list是几个特征的,训练时为5, 预测时为1
        output: data shape:[C, B - D - 1, D]，e.g. [5, 118, 5]
    '''
    def __data_preprocess__(self, data_list, channel_num):
        N = data_list.shape[1] 
        all_channel_data = []

        for c in range(channel_num):
            data = []
            per_channel_data_list = data_list[c]
            for day in range(0, N - self.days - 2):
                data.append(np.log(per_channel_data_list[day + 1: day + self.days + 2]) - 
                                np.log(per_channel_data_list[day: day + self.days + 1]))
            
            all_channel_data.append(data)

        return np.array(all_channel_data)

    '''
        使用 (d1, d2, d3, d4, d5)的类别生成唯一的key, 和对应概率
        其中 key = str(d1 d2 d3 d4 d5)， prob = p1 * p2 * p3 * p4 * p5
        e.g. 如类别 1 2 3 4 5, key = "12345"
    '''
    def __key__(self, x, p):
        prob = 1
        key = ""

        for i in range(x.shape[0]):
            key += str(x[i])
            prob *= p[i]

        return {"key": key, "prob": prob}

    '''
        获取类别和对应的概率，假设L个类别， 则有 L - 1 段, 给定x即可通过 (x - min_x) / interval 得到在第几段，分别上下取整得到l1, l2类别
        而对应的概率为 abs(x - (min_x + l1 * interval)) / interval 和 abs(x - (min_x + l2 * interval)) / interval
        超出 min_x 和 max_x 的概率直接设为1

        input:  data: [B, D]
                channel: index of feature
        output: label: [B, D, 2]
                label_p: [B, D, 2], 1 means out of range[xmin, xmax]
    '''
    def __get_labels__(self, data, channel):
        B = data.shape[0]
        min_x = self.range[channel, :, 0]
        max_x = self.range[channel, :, 1]
        interval = (max_x - min_x) / (self.labels - 1)
       
        label_floor = np.floor((data - min_x.reshape(1, self.days).repeat(B, 0)) /  interval.reshape(1, self.days).repeat(B, 0)).astype(np.int32)
        label_ceil = np.ceil(((data - min_x.reshape(1, self.days).repeat(B, 0)) /  interval.reshape(1, self.days).repeat(B, 0))).astype(np.int32)
        
        label = np.stack((label_floor, label_ceil), 2)
        
        label_p = abs(min_x.repeat(2, 0).reshape(1, 2 * self.days).repeat(B, 0).reshape(B, self.days, 2) + 
                    label * interval.repeat(2, 0).reshape(1, 2 * self.days).repeat(B, 0).reshape(B, self.days, 2) - 
                    data.repeat(2, 1).reshape(B, self.days, 2)) / interval.repeat(2, 0).reshape(1, 2 * self.days).repeat(B, 0).reshape(B, self.days, 2)

        label_p = np.where( (label <= 0)|(label >= self.labels - 1), 1 * np.ones((B, self.days, 2)), label_p)
        label = np.clip(label, 0, self.labels - 1)

        return label, label_p
    
    '''
        对一个batch进行预测， 对于至多 2^D = 32 个类别我们都进行计算，加权平均最后的结果，这点和训练时使用概率最大类别不同
        input:  data: [B, D]
                channel: feature index
        output: data: [B, 1]
    '''
    def __predict_batch__(self, data, channel):
        result = []
        B = data.shape[0]

        label, label_p = self.__get_labels__(data, channel)

        for b in range(B):
            y = 0
            for k in range(1 << self.days):
                label_tmp = []
                label_p_tmp = []
                for d in range(self.days):
                    label_tmp.append(label[b, d, (k >> d) & 0x1])
                    label_p_tmp.append(label_p[b, d, (k >> d) & 0x1])

                key_prob = self.__key__(
                                np.array(label_tmp),
                                np.array(label_p_tmp),
                            )
                key = key_prob["key"]
                
                if key in self.mapping[channel].keys():
                    #print(key)
                    y += self.mapping[channel][key][2] * key_prob["prob"]
            
            # if(y == 0):
            #     print(label[b, :, :])
            #     print(1 / 0)
            result.append(y)

        return np.array(result)

    '''
        对一个batch进行训练：更新映射关系和范围
        input:  data: [B, D]
                channel: feature index
    '''
    def __train_batch__(self, data, channel):
        B = data.shape[0]

        min_x = data.min(0)[:-1]
        max_x = data.max(0)[:-1]
    
        # update the range info
        self.range[channel, :, 0] = min_x
        self.range[channel, :, 1] = max_x

        # compute labels
        label, label_p = self.__get_labels__(data[:, :-1], channel)
        
        # 求最大概率类别
        max_index = np.argmax(label_p, 2)
        label = label[np.arange(label.shape[0])[:,None], np.arange(label.shape[1]), max_index] 
        label_p = label_p[np.arange(label_p.shape[0])[:,None], np.arange(label_p.shape[1]), max_index] 
        
        #print(label[0])
        #print(label_p)
        # update mapping
        for b in range(B):
            key_prob = self.__key__(label[b], label_p[b])
            key = key_prob["key"]
            if not key in self.mapping[channel].keys():
                self.mapping[channel][key] = [0, 0, 0]
            self.mapping[channel][key][0] += key_prob["prob"] * data[b][-1]
            self.mapping[channel][key][1] += key_prob["prob"]
            self.mapping[channel][key][2] += self.mapping[channel][key][0] / self.mapping[channel][key][1]
        #print(self.mapping[channel])
        #print(1 / 0)
    '''
        开始训练，每次取batch个数据进行训练
        第一步使用 __data_preprocess__ 将[C, B]的数据变成时序的[C, B, D]
             同时提取 指标0 的y作为全部的真实值(gt)
        第二步
            针对5个指标，进行第一层训练，得到5个[B, 1]的结果，将结果拼接为[B, 5](second_layer_data)
        第三步
            将 second_layer_data 再次进行训练
    '''
    def train(self, clear_per_batch=False):
        #print("train")
        avg_correct_rate = 0
        epoch = 0
        for i in range(self.data[0].shape[0], self.batch_size + 1, -1):
            epoch += 1
            resize_data = self.__data_preprocess__(self.data[:, i - self.batch_size: i], self.channel)
            
            gt = resize_data[0][:, -1]
            gt = gt.reshape(gt.shape[0], 1)
            
            second_layer_data = []
            for c in range(self.channel):
                # 拼接真实值
                data = np.concatenate([resize_data[c, :, :-1], gt], 1)
                # 训练一个batch
                self.__train_batch__(data, c)
                # 预测一个batch的结果
                yi = self.__predict_batch__(data[:,:-1], c)
                second_layer_data.append(yi)
            
            second_layer_data = np.array(second_layer_data).transpose(1, 0)
            second_layer_data = np.concatenate([second_layer_data, gt], 1)
            self.__train_batch__(second_layer_data, self.channel)
            
            # 计算 Loss
            #print(second_layer_data[:,:-1])
            count = 0
            est_y = self.__predict_batch__(second_layer_data[:,:-1], self.channel)
            est_y = est_y.squeeze()
            gt = gt.squeeze()
            sign = est_y * gt
    
            for _ in sign:
                if _ > 0:
                    count += 1
            count = 100 * count / sign.shape[0]
            # print(gt)
            # print(est_y)
            #print(count)
            #print(1 / 0)
            avg_correct_rate += count
            print("{} epochh, correct rate: {}%".format(epoch, count))
            if clear_per_batch:
                self.mapping = [{}] * (self.channel + 1)
        print("avg correct rate: {}%".format(avg_correct_rate / epoch))

    '''
        当一个新数据来后，预测，先跑第一层，再跑第二层
    '''
    def predict(self, one_data):
        second_layer_data = []
        for c in range(self.channel):
            yi = self.__predict_batch__(one_data[c], c)
            second_layer_data.append(yi)
        
        second_layer_data = np.array(second_layer_data).transpose()
        y = self.__predict_batch__(second_layer_data, self.channel)
        return y

if __name__ == '__main__':
    #base_dir = "/Users/shaochangxu/scx/work/MeiLi/"
    start_time = time.time()
    base_dir = "./"

    data1 = read_data(base_dir + "hkhsi.xlsx", 5)[1:]
    data2 = read_data(base_dir + "hk0005.xlsx", 5)[1:]
    data3 = read_data(base_dir + "hk0388.xlsx", 5)[1:]
    data4 = read_data(base_dir + "hk0700.xlsx", 5)[1:]
    data5 = read_data(base_dir + "hk0016.xlsx", 5)[1:]

    input = np.array([data1, data2, data3, data4, data5]) # 5 * 3370
    # print(input.shape)
    fp = FuzzyPredictor(input, batch_size=125)
    fp.train(clear_per_batch=True)

    test_input = np.array([[
                            [14237.41992, 14045.90039, 13764.36035, 13712.04004, 13574.86035],
                            [79.765144, 79.163147, 78.561142, 78.561142, 77.959152],
                            [13.745107, 13.548278, 13.384255, 13.121819, 13.023405],
                            [0.835453, 0.867064, 0.849001, 0.849001, 0.849001],
                            [52.073238, 51.06863, 49.39423, 49.39423, 48.054733]
    ]])

    test_input = test_input.transpose(1, 0, 2)
    re = fp.predict(test_input)
    if re > 0:
        print("涨")
    else:
        print("跌")
    end_time = time.time()
    print("run time is: {}%".format(end_time - start_time))