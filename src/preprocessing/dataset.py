import torch
import numpy as np
from torch.utils.data import Dataset
import os
import functools
import bisect 
import random 

# Cache lại các file npy
@functools.lru_cache(maxsize=16) 
def get_mmap_array(file_path):
    try:
        return np.load(file_path, mmap_mode='r')
    except Exception as e:
        print(f"Lỗi mmap file băm {file_path}: {e}")
        return None

class ChessSLDataset(Dataset):
    """
    Tự chia train/val và tự shuffle các file băm (chunk).
    """
    
    def __init__(self, processed_dir, mode='train', val_split=0.2, shuffle_chunks=True):
        self.processed_dir = processed_dir
        self.shards_info = []
        self.cumulative_sizes = [0]
        
        print(f"\nĐang lập chỉ mục cho mode: '{mode}'...")
        
        try:
            base_names = sorted(
                [f.replace(".X.npy", "") for f in os.listdir(processed_dir) if f.endswith(".X.npy")]
            )
        except FileNotFoundError:
            print(f"Không tìm thấy {processed_dir}")
            raise
        if not base_names:
            print(f"Không có dữ liệu")
            raise

        if shuffle_chunks:
            print(f"  -> Đang xáo trộn (shuffle) {len(base_names)} file băm...")
            random.shuffle(base_names)
        
        # Chia danh sách file băm, KHÔNG chia data bên trong
        split_idx = int(len(base_names) * (1.0 - val_split))
        
        if mode == 'train':
            self.chunk_base_names = base_names[:split_idx]
            print(f"  -> Mode 'train': Lấy {len(self.chunk_base_names)}/{len(base_names)} file băm đầu tiên.")
        else: # mode == 'val'
            self.chunk_base_names = base_names[split_idx:]
            print(f"  -> Mode 'val': Lấy {len(self.chunk_base_names)}/{len(base_names)} file băm cuối cùng.")

        # Quét header (Chỉ quét các file của mode này)
        print("  -> Đang quét header của các file băm được chọn...")
        for base_name in self.chunk_base_names:
            path_X = os.path.join(processed_dir, f"{base_name}.X.npy")
            
            mmap_temp = None
            try:
                mmap_temp = np.load(path_X, mmap_mode='r')
                length = len(mmap_temp)
                
                if length > 0:
                    self.shards_info.append({'base_name': base_name, 'length': length})
                    self.cumulative_sizes.append(self.cumulative_sizes[-1] + length)
                
            except Exception as e:
                print(f"Lỗi khi đọc header {path_X}: {e}. Bỏ qua.")
            finally:
                if mmap_temp is not None:
                    if hasattr(mmap_temp, 'close'): mmap_temp.close()
                    del mmap_temp

        self.total_length = self.cumulative_sizes[-1]
        print(f"  -> Tổng cộng {self.total_length} mẫu cho mode '{mode}'.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")
        shard_index = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[shard_index]
        base_name = self.shards_info[shard_index]['base_name']
        base_path = os.path.join(self.processed_dir, base_name)
        X_mmap = get_mmap_array(f"{base_path}.X.npy")
        P_mmap = get_mmap_array(f"{base_path}.y_policy.npy")
        V_mmap = get_mmap_array(f"{base_path}.y_value.npy")
        if X_mmap is None or P_mmap is None or V_mmap is None:
            raise IOError(f"Không thể đọc data từ {base_path}")
        X_item = X_mmap[local_idx]
        P_item = P_mmap[local_idx]
        V_item = V_mmap[local_idx]
        return (
            torch.from_numpy(X_item.astype(np.float32)),
            torch.tensor(P_item, dtype=torch.long),
            torch.tensor(V_item, dtype=torch.float32).unsqueeze(0)
        )