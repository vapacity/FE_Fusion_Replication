import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from experiment_loadData import DatabaseDataset
from torch.nn import CosineSimilarity
import torch
import torchvision.transforms as transforms
import FE_Net
from torch.utils.data import ConcatDataset

database_dirs = ['/root/autodl-tmp/processed_data/dvs_vpr_2020-04-21-17-03-03']
gps_files = ['/root/autodl-tmp/processed_data/dvs_vpr_2020-04-21-17-03-03/interpolated_gps.txt']
# # 数据库的路径和时间戳文件
# database_dir = '/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03'
# gps_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-24-15-12-03/interpolated_gps.txt"
# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# 生成特征的函数，输出为 torch.Tensor
def generate_features(data_loader, model):
    features = []
    gps_data_list = []  # 用来存储GPS信息
    with torch.no_grad():
        for frames, event_volumes, gps_info in tqdm(data_loader, desc="Generating features"):
            batch_size = frames.size(0)
            
            # 确保数据在GPU上
            frames, event_volumes = frames.cuda(), event_volumes.cuda()
            
            # 提取特征，保持为 torch.Tensor（在 GPU 上）
            feature_batch = model(frames, event_volumes)  # 提取特征，仍然在GPU上
            
            # 将特征添加到列表
            features.append(feature_batch)
            
            # 将对应的 GPS 信息保存
            gps_data_list.extend(gps_info)  # gps_info 是每个样本的 GPS 数据，可能是列表或元组

    # 合并所有批次的特征，将其保持为 torch.Tensor
    features_tensor = torch.cat(features, dim=0)  # 按照第 0 维度（批次）拼接所有特征
    
    return features_tensor# 返回特征张量和对应的GPS数据

# database_dataset = DatabaseDataset(database_dir, gps_file, transform)
# database_loader = DataLoader(database_dataset, batch_size=8, shuffle=False, num_workers=1)

# 数据集实例
# 创建多个数据集实例
datasets = [DatabaseDataset(dir, gps, transform) for dir, gps in zip(database_dirs, gps_files)]

# 使用 ConcatDataset 组合它们
combined_dataset = ConcatDataset(datasets)

# 创建 DataLoader
database_loader = DataLoader(combined_dataset, batch_size=8, shuffle=False, num_workers=1)


# 加载模型
model = FE_Net.MainNet(channel_sizes=[128, 256, 512]).cuda()
model.load_state_dict(torch.load("/root/FE_Fusion/train/result_2024-10-26-15-18/saved_model/model_epoch_60.pth"))
model.eval()  # 设置模型为评估模式

# 生成数据库特征并保存
database_features = generate_features(database_loader, model)
print("database features len:",len(database_features))

def recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, N=5, distance_threshold=75):
    total_queries = 0
    correct_count = 0  # 记录检索正确的次数
    global_idx = 0

    database_features = torch.tensor(database_features).cuda()

    with torch.no_grad():
        for frames, event_volumes, query_gps_info in query_loader:
            frames, event_volumes = frames.cuda(), event_volumes.cuda()
            query_features = model(frames, event_volumes)  # 提取 query 特征
            
            cos = CosineSimilarity(dim=2,eps=1e-8)

            similarity_scores = cos(query_features.unsqueeze(1),database_features.unsqueeze(0))

            top_n_scores, top_n_indices = torch.topk(similarity_scores, N, dim=1, largest=True, sorted=True)

            # 对每个查询批次的样本检查 Recall@N
            for i in range(query_features.size(0)):  # 遍历 batch 中的每个 query
                query_lat, query_lon = query_gps_data[global_idx][0], query_gps_data[global_idx][1]  # 当前查询样本的GPS坐标
                # print("query:",global_idx,query_lat,query_lon)
                for idx in top_n_indices[i]:  # 遍历 top N 的索引
                    db_lat, db_lon = database_gps_data[idx.item()][0], database_gps_data[idx.item()][1]  # 数据库样本的GPS坐标
                    # 计算地理距离
                    # print("database:",idx,db_lat,db_lon)
                    distance = geodesic((query_lat, query_lon), (db_lat, db_lon)).meters

                    if distance < distance_threshold:
                        # print("yes")
                        correct_count += 1
                        break  # 找到一个正确的就跳出
                global_idx += 1

            total_queries += query_features.size(0)  # 每次处理 batch_size 个查询样本

    recall = correct_count / total_queries
    print(f"Recall@{N}: {recall:.4f}")
    return recall


# # 读取之前保存的数据库特征和GPS数据
# database_features = np.load('/root/autodl-tmp/experiment/database_features.npy')
# database_gps_data = np.load('/root/autodl-tmp/experiment/database_gps_data.npy', allow_pickle=True)

# 假设查询集路径
query_dir = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-22-17-24-21"
query_gps_file = "/root/autodl-tmp/processed_data/dvs_vpr_2020-04-22-17-24-21/interpolated_gps.txt"

# 构建查询数据集
query_dataset =DatabaseDataset(query_dir, query_gps_file, transform)
query_loader = DataLoader(query_dataset, batch_size=8, shuffle=False, num_workers=2)

database_gps_data = []
for gps_file in gps_files:
    with open(gps_file, 'r') as f:
        for line in f:
            # 假设每行的格式是：lat lon timestamp
            lat, lon, timestamp = line.strip().split()
            
            # 将 (lat, lon) 作为 tuple 存储
            database_gps_data.append((float(lat), float(lon)))

query_gps_data = []
with open(query_gps_file, 'r') as f:
    for line in f:
        # 假设每行的格式是：lat lon timestamp
        lat, lon, timestamp = line.strip().split()
        
        # 将 (lat, lon) 作为 tuple 存储
        query_gps_data.append((float(lat), float(lon)))

# 计算 Recall@N
recall_at_5 = recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, N=1, distance_threshold=75)
recall_at_5 = recall_at_n_with_distance(query_loader, database_features, query_gps_data, database_gps_data, model, N=5, distance_threshold=75)
