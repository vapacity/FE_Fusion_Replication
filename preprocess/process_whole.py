import os
import h5py
#from process_event import process_event
#from process_frame import process_frame
#from read_nmea import process_gps
from tqdm import tqdm
import numpy as np
from geopy.distance import geodesic

'''
dvs_vpr_2020-04-24-15-12-03
dvs_vpr_2020-04-27-18-13-29
dvs_vpr_2020-04-28-09-14-11
dvs_vpr_2020-04-29-06-20-23
'''

def read_gps_data(gps_file_path):
    return np.loadtxt(gps_file_path, delimiter=' ')

def interpolate_gps_data(event_file_path,gps_file_path):
    gps_data = read_gps_data(gps_file_path)
    gps_timestamps = gps_data[:, 2]
    gps_idx = 0
    count = 0
    with h5py.File(event_file_path, 'a') as f:  # 'a' 模式用于追加数据
            total_groups = len(f.keys())
            
            for group_name in tqdm(f.keys(), desc='processing gps_data', total=total_groups):
                group = f[group_name]
                event_time = group['timestamp'][()]
                if event_time < gps_timestamps[0]:
                    continue

                while gps_idx < len(gps_timestamps) - 1 and event_time > gps_timestamps[gps_idx + 1]:
                    gps_idx += 1

                if gps_idx == 0 or gps_idx == len(gps_timestamps) - 1:
                    # Cannot interpolate at the beginning or end
                    interpolated_lat, interpolated_lon = None, None
                else:
                    # Perform linear interpolation
                    t0, t1 = gps_timestamps[gps_idx], gps_timestamps[gps_idx + 1]
                    lat0, lat1 = gps_data[gps_idx, 0], gps_data[gps_idx + 1, 0]
                    lon0, lon1 = gps_data[gps_idx, 1], gps_data[gps_idx + 1, 1]
                    alpha = (event_time - t0) / (t1 - t0)
                    interpolated_lat = lat0 + alpha * (lat1 - lat0)
                    interpolated_lon = lon0 + alpha * (lon1 - lon0)

               # 将插值后的 GPS 数据写入现有组
                if interpolated_lat is not None and interpolated_lon is not None:
                    if 'interpolated_lat' in group:
                        del group['interpolated_lat']
                        del group['interpolated_lon']
                    count += 1
                    group.create_dataset('interpolated_lat', data=interpolated_lat)
                    group.create_dataset('interpolated_lon', data=interpolated_lon)
    print(count)
event_path = '/root/autodl-tmp/processed_data/event_output/dvs_vpr_2020-04-24-15-12-03.h5'
gps_path = '/root/autodl-tmp/processed_data/gps_output/20200424_151015-daytime_concat.txt'
interpolate_gps_data(event_path,gps_path)

def read_hdf5_gps_data(h5_file_path):
    gps_data = []

    with h5py.File(h5_file_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            if 'interpolated_lat' in group and 'interpolated_lon' in group:
                interpolated_lat = group['interpolated_lat'][()]
                interpolated_lon = group['interpolated_lon'][()]
                gps_data.append((group_name, interpolated_lat, interpolated_lon))
        
    return gps_data
# 示例调用
gps_data = read_hdf5_gps_data(event_path)
print(f"读取到 {len(gps_data)} 条 GPS 数据")

def find_triplet_samples(gps_data, pos_threshold=25, neg_threshold=75):
    triplet_samples = []

    for anchor_idx, anchor in enumerate(tqdm(gps_data, desc="Finding triplet samples")):
        anchor_name, anchor_lat, anchor_lon = anchor
        pos_samples = []
        neg_samples = []
        
        for sample_idx, sample in enumerate(gps_data):
            sample_name, sample_lat, sample_lon = sample
            if anchor_name == sample_name:
                continue

            distance = geodesic((anchor_lat, anchor_lon), (sample_lat, sample_lon)).meters

            if distance < pos_threshold:
                pos_samples.append(sample_idx)
            elif distance > neg_threshold:
                neg_samples.append(sample_idx)
        
        if pos_samples and len(neg_samples) >= 10:
            positive_sample = np.random.choice(pos_samples)
            negative_samples = np.random.choice(neg_samples, 10, replace=False)
            triplet_samples.append((anchor_idx, positive_sample, negative_samples))
    
    return triplet_samples

# 示例调用
triplet_samples = find_triplet_samples(gps_data)
print(triplet_samples)
print(f"找到 {len(triplet_samples)} 个三元组样本")

def save_triplet_samples_to_hdf5(h5_file_path, gps_data,triplet_samples):
    with h5py.File(h5_file_path, 'a') as f:
        for anchor_idx, positive_sample, negative_samples in triplet_samples:
            group_name = gps_data[anchor_idx][0]
            group = f[group_name]
            if 'positive' in group:
                del group['positive']
            if 'negative' in group:
                del group['negative']
            group.create_dataset('positive', data=positive_sample)
            group.create_dataset('negative', data=negative_samples)
            print("anchor:",group['timestamp'])
            print("pos:",f[positive_sample]['timestamp'])
            print("neg:",f[negative_samples]['timestamp'])

# 示例调用
save_triplet_samples_to_hdf5(event_path,gps_data, triplet_samples)



bag_file = '/root/autodl-fs/Brizbane_dataset/dvs_vpr_2020-04-24-15-12-03.bag'
gps_file = '20200424_151015-daytime_concat'
frame_dir = '/root/autodl-tmp/processed_data/frame_output/'
gps_list = []

h5_file_path = '/root/autodl-tmp/processed_data/event_output/dvs_vpr_2020-04-29-06-20-23.h5'
gps_file_path = '/root/autodl-tmp/processed_data/gps_output/20200429_061912-sunrise_concat.txt'

#gps_data = read_hdf5_gps_data(h5_file_path)
#triplet_samples=find_triplet_samples(gps_data)
#save_triplet_samples_to_hdf5(h5_file_path,triplet_samples)

#inspect_hdf5_file(file_path)
#print('process_frame')
#process_frame(bag_file, frame_dir, frame_interval=0.25)
#print('process_event')
#events = process_event(bag_file, frame_dir, time_tolerance = 0.0125)
#print('process_gps')
#gps_data = process_gps(gps_file)
#print("interpolate")

#gps_list = interpolate_gps_data(h5_file_path, gps_file_path)

