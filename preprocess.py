import numpy as np
import pandas
import pandas as pd
import torch
import os


def downsample(data, max_length=100):
    sample_step = (data.shape[0] // max_length)+1
    return data[::sample_step]


# 根据label_df的(起始时间，终止时间)筛选出data_df中对应的行, 以(X,y)的形式返回数据
def match_data_label(data_df, label_df, label_file_name):
    tensors = []
    # 处理时间
    data_df['ISO time'] = pd.to_datetime(data_df['ISO time'],
                                         format="%H:%M:%S", exact=False)
    label_df['起始时间'] = pd.to_datetime(label_df['起始时间'], format="%H:%M:%S")
    if '终止时间' in label_df.columns:  # 有的文件使用'结束时间'表述终止时间
        label_df['终止时间'] = pd.to_datetime(label_df['终止时间'], format="%H:%M:%S")
    else:
        label_df['终止时间'] = pd.to_datetime(label_df['结束时间'], format="%H:%M:%S")
    for _, row in label_df.iterrows():  # 遍历标签文件，在数据文件中找到对应的部分
        start_time, end_time = row['起始时间'], row['终止时间']
        data = data_df[(data_df['ISO time'] >= start_time) & (data_df['ISO time'] <= end_time)]
        data = data[['Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw']].values
        # data = data[['Longitude', 'Latitude', 'Altitude']].values
        if data.shape[0] == 0: # 异常数据
            print(f"异常文件：{label_file_name}, 起始时间：{start_time}, 终止时间：{end_time}")
        else:
            max_length = 100
            if data.shape[0]>max_length:
                print(f"数据过长：{label_file_name}, 起始时间：{start_time}, 终止时间：{end_time}"
                      f"长度：{data.shape[0]}")
                data = downsample(data,max_length=max_length)
                print(data.shape)
            # 转为tensor形式返回
            X = torch.tensor(data)
            if torch.any(torch.isnan(X)):
                print(f"含nan：{label_file_name}, 起始时间：{start_time}, 终止时间：{end_time}")
            y = torch.tensor(int(label_file_name.split('-')[1])-1)   # A-i-XXX-XX转为标签值i-1
            tensors.append([X,y])
    return tensors


# 处理文件夹内的数据
def process_data(path):
    tensors = []
    data_files = []
    label_files = []
    # 遍历数据文件夹
    for _, _, files in os.walk(path):
        for file in files:
            # 数据文件
            if file.endswith('csv'):
                data_files.append(file)
            # 标签文件
            if file.endswith('.xls'):
                label_files.append(file)
    # 处理数据文件
    data_df = pd.DataFrame()
    for data_file in data_files:
        df = pd.read_csv(os.path.join(path, data_file))
        data_df = pd.concat([data_df, df])
        for col in ['Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw']:
            data_df[col] = data_df[col].interpolate(method='linear')
        for col in ['Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw']:
            x = data_df[col]
            # print(any(pd.isna(x)))
            data_df[col] = (x - x.mean()) / (x.std() + 1e-5)

        # 寻找数据文件同名的标签文件
        label_file_name = data_file.replace('csv', 'xls')
        if label_file_name in label_files:
            label_df = pd.read_excel(os.path.join(path, label_file_name))

            tensors.extend(match_data_label(data_df, label_df, label_file_name))
        else:  # 如果找不到，说明是05!07, 13!14的其中一种，单独处理
            # print("找不到：",label_file_name)
            old_label_file_name = label_file_name
            for maneuver in ['05','07','13','14']:
                label_file_name = old_label_file_name[:2] + maneuver + old_label_file_name[7:]
                label_file_path = os.path.join(path, label_file_name)
                # print('尝试:',label_file_path)
                if os.path.exists(label_file_path):
                    # print('找到:', label_file_path)
                    label_df = pd.read_excel(label_file_path)
                    tensors.extend(match_data_label(data_df, label_df, label_file_name))
    return tensors


# 遍历路径下的所有文件夹，处理数据, tensors是一个由(数据, 标签)组成的列表
def traverse(root_path):
    tensors = []
    dirs = os.listdir(root_path)
    for dir in dirs:
        dir_path = os.path.join(root_path, dir)
        tensors.extend(process_data(dir_path))
    X = [tensor[0] for tensor in tensors]
    pad_X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    y = torch.tensor([tensor[1] for tensor in tensors])
    tensors = [pad_X, y]
    import collections
    iy = [int(i) for i in y]
    c = collections.Counter(iy)
    print(c)
    print(pad_X.shape, y.shape)
    return tensors


if __name__ == '__main__':
    root_path = r'F:\data\plane\A'  # 数据路径
    tensors = traverse(root_path)