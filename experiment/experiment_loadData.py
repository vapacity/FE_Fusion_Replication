import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from geopy.distance import geodesic
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class DatabaseDataset(Dataset):
    def __init__(self, database_dir, gps_file, transform=None):
        """
        Args:
            database_dir (str): 数据库的根目录，包含 'frame' 和 'event' 子文件夹。
            gps_file (str): 存储GPS数据的文件路径，包含纬度、经度和时间戳。
            transform (callable, optional): 应用于帧图像的变换操作。
        """
        self.database_dir = database_dir
        self.transform = transform
        
        # 从 GPS 文件中读取时间戳和 GPS 信息
        self.gps_data, self.timestamps = self._read_gps_file(gps_file)
        
        if len(self.timestamps) == 0:
            raise ValueError("No valid data found in the dataset.")

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        # 获取时间戳
        timestamp = self.timestamps[idx]
        
        # 加载帧图像和事件体数据
        frame = self._load_frame(self.database_dir, timestamp)
        event_volume = self._load_event_volume(self.database_dir, timestamp)
        
        # 获取 GPS 信息
        gps_info = self.gps_data[timestamp]
        #print(gps_info)
        if gps_info is None:
            print(f"Missing GPS info for timestamp {timestamp}")
            return None  # 如果GPS信息缺失，返回None

        return frame, event_volume, gps_info

    def _load_frame(self, dir, timestamp):
        """
        加载帧图像
        Args:
            dir (str): 数据库路径。
            timestamp (str): 帧图像的时间戳。
        Returns:
            Image: 加载的图像。
        """
        frame_path = os.path.join(dir, "frame", f"{timestamp}.png")
        frame = Image.open(frame_path).convert('L')  # 转换为灰度图
        #print(frame_path)
        if self.transform:
            frame = self.transform(frame)
        return frame

    def _load_event_volume(self, dir, timestamp):
        """
        加载事件体数据
        Args:
            dir (str): 数据库路径。
            timestamp (str): 事件体数据的时间戳。
        Returns:
            Tensor: 加载的事件体数据。
        """
        event_path = os.path.join(dir, "event", f"{timestamp}.npy")
        #print(event_path)
        event_volume = np.load(event_path)
        event_volume = torch.tensor(event_volume).float()
        event_volume = torch.nn.functional.interpolate(event_volume.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        return event_volume

    def _read_gps_file(self, gps_file):
        """
        读取GPS文件，返回包含时间戳和GPS信息的字典。
        
        Args:
            gps_file (str): GPS 文件路径，包含纬度、经度和时间戳。
            
        Returns:
            gps_data (dict): GPS 数据字典，键为时间戳，值为 (latitude, longitude)。
            timestamps (list): 时间戳列表，用于加载帧图像和事件体数据。
        """
        gps_data = {}
        timestamps = []
        
        with open(gps_file, 'r') as file:
            for line in file:
                lat, lon, timestamp = line.strip().split()
                gps_data[timestamp] = (float(lat), float(lon))
                timestamps.append(timestamp)
        
        return gps_data, timestamps


