import numpy as np 
import os
import glob 
import config.config as config
import gc
from tqdm import tqdm


OUTPUT_X = os.path.join(config.MERGE_PROCESSED_DIR, "X.npy")
OUTPUT_P = os.path.join(config.MERGE_PROCESSED_DIR, "y_policy.npy")
OUTPUT_V = os.path.join(config.MERGE_PROCESSED_DIR, "y_value.npy")

SUFFIXES_TO_MERGE = [
    ("X.npy", OUTPUT_X),
    ("Y_policy.npy", OUTPUT_P),
    ("Y_value.npy", OUTPUT_V)
]

if __name__ == "__main__": 
    print("Bắt đầu gộp file")
    os.makedirs(config.MERGE_PROCESSED_DIR, exist_ok=True)
    for suffix, output_file in SUFFIXES_TO_MERGE: 
        file_paths = []
        for source in config.SOURCES: 
            search_pattern = os.path.join(os.path.join(config.PROCESSED_DIR, source), f"*.{suffix}")
            file_paths.extend(glob.glob(search_pattern))

        if not file_paths: 
            print(f"Không tìm thấy file mẫu: {suffix}")
        
        print(f"\nĐang xử lý {len(file_paths)} file '{suffix}'")
        total_length = 0 
        first_file = True # file đầu để xác định shape, kiểu dữ liệu 
        dtype = None 
        item_shape = None 
        
        for path in file_paths: 
            try: 
                mmap_arr = np.load(path, mmap_mode="r")
                total_length += len(mmap_arr)
                if first_file: 
                    item_shape = mmap_arr.shape[1:]
                    dtype = mmap_arr.dtype
                    first_file = False 
            except Exception as e: 
                print(f"     Lỗi {e} khi đọc {path}. Bỏ qua!")

        if total_length == 0: 
            print("Không có dữ liệu để gộp")
            continue

        final_shape = (total_length, ) + item_shape
        print(f"   -> Shape: {final_shape}")
        print(f"   -> Kiểu dữ liệu: {dtype}")

        print(f"Tạo file rỗng (mmap) tại {output_file}")  
        try: 
            total_mmap = np.lib.format.open_memmap(
                output_file, 
                mode="w+",
                dtype=dtype,
                shape=final_shape
            )
        except Exception as e: 
            print(f"     Lỗi trong quá trình tạo file mmap: {e}")
            continue

        print(f"Đang copy dữ liệu...")
        current_index = 0 
        cop_bar = tqdm(file_paths, desc="Đang copy", unit="file")
        for path in cop_bar: 
            try: 
                cop_bar.postfix(f"copy {os.path.basename(path)}")
                chunk_arr = np.load(path)
                chunk_len = len(chunk_arr)

                total_mmap[current_index : current_index + chunk_len] = chunk_arr 
                current_index += chunk_len

                # Giải phóng ram
                # gc.collect()
                del chunk_arr

            except Exception as e: 
                print(f"Lỗi khi copy file {os.path.basename(path)}. Bỏ qua!")
                continue

        print(f"Đang lưu vào {output_file}")
        total_mmap.flush()
        del total_mmap
        # gc.collect()
        print(f"Gộp thành công {output_file}")

    print("Quá trình gộp hoàn tất!!")