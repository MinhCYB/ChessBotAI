from collections import deque
import torch
from torch.utils.data import Dataset # <-- IMPORT MỚI
import random
import numpy as np

# class ReplayBuffer(Dataset): # <-- KẾ THỪA TỪ DATASET
class ReplayBuffer(Dataset):
    def __init__(self, buffer_size_samples: int):
        self.buffer = [] 
        self.maxlen = buffer_size_samples
    
    def add_game_data(self, game_data: list):
        """
        Thêm data (list of tuples) của một ván cờ vào buffer.
        """
        self.buffer.extend(game_data)
        
        # Cắt bớt phần cũ nếu buffer bị đầy
        if len(self.buffer) > self.maxlen:
            # Giữ lại N phần tử cuối cùng
            self.buffer = self.buffer[-self.maxlen:] 

    def __len__(self):
        # Trả về độ dài của buffer
        return len(self.buffer)
    
    def __getitem__(self, idx):
        """
        Hàm này được DataLoader gọi (với num_workers=0).
        Nó chỉ lấy 1 MẪU và ép kiểu "lười".
        """
        # Lấy 1 MẪU từ list (siêu nhanh)
        # (state, policy, value) vẫn đang là numpy/int/float thô
        state, policy, value = self.buffer[idx] 
        
        # Ép kiểu "lười" y hệt Giai đoạn 1
        # (Chuyển đổi ở đây tốn rất ít RAM)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32) # Thêm chiều
        )