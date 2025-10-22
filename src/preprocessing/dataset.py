# src/preprocessing/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import functools
import bisect # Thư viện để tìm kiếm nhị phân (siêu nhanh)

# --- Bộ đệm (Cache) cho file mmap ---
# Cache này sẽ lưu các file mmap đã mở để không phải mở lại
# maxsize=None nghĩa là cache không giới hạn (giữ tất cả file handles)
# Đây là chìa khóa để giải quyết WinError 8 mà vẫn nhanh
@functools.lru_cache(maxsize=None)
def get_mmap_array(file_path):
    """
    Hàm helper: Mở file .npy ở chế độ mmap và cache lại.
    Hàm này chỉ thực sự chạy 1 LẦN cho mỗi file_path.
    """
    try:
        return np.load(file_path, mmap_mode='r')
    except Exception as e:
        print(f"LỖI nghiêm trọng khi mmap file {file_path}: {e}")
        return None

class ChessSLDataset(Dataset):
    """
    Dataset "siêu lười": quét thư mục để lấy "mục lục"
    và chỉ mở file khi __getitem__ được gọi.
    """
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.file_base_paths = []
        self.cumulative_sizes = [0] # Mục lục
        
        print(f"Đang lập chỉ mục (indexing) thư mục: {processed_dir}")
        
        file_names = sorted(
            [f for f in os.listdir(processed_dir) if f.endswith(".X.npy")]
        )
        
        for file_name in file_names:
            base_name = file_name.replace(".X.npy", "")
            base_path = os.path.join(self.processed_dir, base_name)
            
            # Mở file .X.npy TẠM THỜI chỉ để lấy độ dài
            try:
                # Dùng mmap_mode='r' để đọc header nhanh
                X_mmap = np.load(f"{base_path}.X.npy", mmap_mode='r')
                file_length = len(X_mmap)
                del X_mmap # Đóng file mmap tạm thời
                
                if file_length > 0:
                    self.file_base_paths.append(base_path)
                    # Thêm vào mục lục
                    self.cumulative_sizes.append(self.cumulative_sizes[-1] + file_length)
                    print(f"  -> Lập chỉ mục '{base_name}' với {file_length} mẫu.")
            except Exception as e:
                print(f"  -> Lỗi khi lập chỉ mục {base_name}: {e}. Bỏ qua.")

        self.total_length = self.cumulative_sizes[-1]
        print(f"Tổng cộng lập chỉ mục {self.total_length} mẫu từ {len(self.file_base_paths)} bộ file.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")

        # 1. Dùng "mục lục" (bisect) để tìm xem idx thuộc file nào
        # (Siêu nhanh, O(logN))
        file_index = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        
        # 2. Tìm index cục bộ (local_idx) bên trong file đó
        local_idx = idx - self.cumulative_sizes[file_index]
        
        # 3. Lấy base_path của file
        base_path = self.file_base_paths[file_index]
        
        # 4. Lấy data từ các file mmap (dùng cache)
        # Các hàm này sẽ chỉ thực sự mở file 1 LẦN
        X_mmap = get_mmap_array(f"{base_path}.X.npy")
        P_mmap = get_mmap_array(f"{base_path}.y_policy.npy")
        V_mmap = get_mmap_array(f"{base_path}.y_value.npy")
        
        if X_mmap is None or P_mmap is None or V_mmap is None:
            raise IOError(f"Không thể đọc data từ {base_path}")

        # 5. Lấy đúng 1 mẫu
        X_item = X_mmap[local_idx]
        P_item = P_mmap[local_idx]
        V_item = V_mmap[local_idx]
        
        # 6. Ép kiểu "lười" (giống hệt code cũ)
        return (
            torch.from_numpy(X_item.astype(np.float32)),
            torch.tensor(P_item, dtype=torch.long),
            torch.tensor(V_item, dtype=torch.float32).unsqueeze(0)
        )