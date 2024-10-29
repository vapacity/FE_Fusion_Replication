import os
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm

# 假设 read_gps_file 是一个从文件中读取GPS数据的函数，返回 (latitude, longitude, timestamp) 元组列表
def read_gps_file(file_path):
    """
    读取GPS文件，假设每一行格式为：latitude, longitude, timestamp
    """
    gps_data = []
    with open(file_path, 'r') as file:
        for line in file:
            lat, lon, timestamp = line.strip().split(',')
            gps_data.append((float(lat), float(lon), float(timestamp)))
    return gps_data

def find_triplet_samples(query_gps_path, database_gps_paths, save_path, pos_threshold=10):
    """
    读取 query 和 database 的 GPS 数据，查找每个 query 对应的正样本（符合距离阈值的样本）。
    
    Args:
        query_gps_path (str): query 的 GPS 数据文件路径。
        database_gps_paths (list of str): database 的 GPS 数据文件路径列表。
        save_path (str): 保存正样本结果的文件路径。
        pos_threshold (float): 作为正样本的距离阈值（单位：米）。
    """
    
    # 读取 query 和 database 的 GPS 数据
    query_gps_data = read_gps_file(query_gps_path)
    database_gps_data = []

    # 读取所有的 database 文件路径
    for gps_path in database_gps_paths:
        database_gps_data.extend(read_gps_file(gps_path))

    # 准备存储正样本的列表
    pos_samples = []

    # 遍历 query_gps_data 中的每一个点，查找对应的正样本
    for anchor in tqdm(query_gps_data, desc="Finding positive samples"):
        anchor_lat, anchor_lon, anchor_time = anchor
        positive_found = False  # 标记是否找到正样本

        # 遍历 database_gps_data 中的每一个点
        for sample in database_gps_data:
            sample_lat, sample_lon, sample_time = sample

            # 计算 anchor 和 sample 之间的地理距离
            distance = geodesic((anchor_lat, anchor_lon), (sample_lat, sample_lon)).meters

            # 根据距离判断是否为正样本
            if distance < pos_threshold:
                # 如果距离在正样本阈值内，记录为正样本
                pos_samples.append((anchor_time, sample_time))
                positive_found = True
                break  # 找到正样本后就不再继续查找

        # 如果没有找到正样本，则跳过该 query
        if not positive_found:
            print(f"No positive sample found for query at {anchor_time}")

    # 将结果写入文件
    with open(save_path, 'w') as file:
        for query_time, pos_time in pos_samples:
            line = f"{query_time}, {pos_time}\n"
            file.write(line)

    print(f"Saved positive samples to {save_path}")
    return pos_samples
