import torch
from torch.utils.data import Dataset 

class ReplayBuffer(Dataset):
    def __init__(self, buffer_size_samples: int):
        self.buffer = [] 
        self.maxlen = buffer_size_samples
    
    def add_game_data(self, game_data: list):
        self.buffer.extend(game_data)
        
        if len(self.buffer) > self.maxlen:
            self.buffer = self.buffer[-self.maxlen:] 

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        state, policy, value = self.buffer[idx] 
        
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32) # thêm chiều
        )