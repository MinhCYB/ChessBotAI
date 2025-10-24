import chess
import chess.pgn
import gc
import logging
import numpy as np
import os
from tqdm import tqdm 
from collections import deque
import subprocess, ast
import json
import csv
import pandas as pd
import glob

import config.config as config
import src.utils.utils as utils

def count_total_positions():
    total_positions = 0
    
    # Tìm tất cả file .pgn trong thư mục
    pgn_files = []
    for source in config.SOURCES:
        pgn_files.extend(glob.glob(os.path.join(os.path.join(config.SPLIT_DIR, source), "*.pgn")))
    
    if len(pgn_files) == 0:
        print(f"!!! Không tìm thấy file .pgn nào trong: {config.SPLIT_DIR}")
        return
        
    print(f"Bắt đầu quét {len(pgn_files)} file PGN...")
    

    for file_path in tqdm(pgn_files, desc="Đang đọc file PGN", unit="file"):
        file_name = os.path.basename(file_path)
        # print(f"  -> Đang quét file {i+1}/{len(pgn_files)}: {file_name}")
        
        file_pos_count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            with tqdm(desc=f"Counting {os.path.basename(file_path)}", unit=" games") as pbar:
                while True:
                    try:
                        # Đọc 1 ván
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break # Hết file
                            
                        # Đếm số nước đi (chính là số thế cờ)
                        # Dùng generator để tiết kiệm mem
                        moves_in_game = sum(1 for _ in game.mainline_moves())
                        file_pos_count += moves_in_game
                        pbar.update(1)
                        
                    except Exception as e:
                        # Bỏ qua ván bị lỗi
                        pass
                        
            print(f"     -> File này có {file_pos_count} thế cờ.")
            total_positions += file_pos_count

    print("\n--- QUÉT HOÀN TẤT ---")
    print(f"Tổng cộng có: {total_positions} thế cờ (positions).")
    return total_positions

def parse_and_write(total_positions):
    X_ITEM_SHAPE = ((config.HISTORY_LENGTH * 12 + 8), 8, 8)
    X_DTYPE = np.float32
    P_DTYPE = np.int64
    V_DTYPE = np.float32
    # --- 1. ĐỊNH NGHĨA SHAPE TỔNG ---
    X_shape = (total_positions,) + X_ITEM_SHAPE
    P_shape = (total_positions,) # 1D array
    V_shape = (total_positions,) # 1D array
    
    print(f"Tổng shape X: {X_shape}")
    print("Cảnh báo: Đảm bảo bro có đủ dung lượng ổ cứng!!!")

    # --- 2. TẠO 3 FILE MMAP RỖNG ---
    # Đảm bảo thư mục tồn tại
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    path_X = os.path.join(config.PROCESSED_DIR, "total.X.npy")
    path_P = os.path.join(config.PROCESSED_DIR, "total.y_policy.npy")
    path_V = os.path.join(config.PROCESSED_DIR, "total.y_value.npy")
    
    try:
        total_X_mmap = np.lib.format.open_memmap(path_X, mode='w+', dtype=X_DTYPE, shape=X_shape)
        total_P_mmap = np.lib.format.open_memmap(path_P, mode='w+', dtype=P_DTYPE, shape=P_shape)
        total_V_mmap = np.lib.format.open_memmap(path_V, mode='w+', dtype=V_DTYPE, shape=V_shape)
    except Exception as e:
        print(f"!!! LỖI NGHIÊM TRỌNG khi tạo file mmap: {e}")
        print(">>> Rất có thể bro đã HẾT Ổ CỨNG. Hãy dọn dẹp!")
        return

    # --- 3. QUÉT PGN LẦN 2 VÀ GHI DỮ LIỆU ---
    current_index = 0
    pgn_files = []
    for source in config.SOURCES:
        pgn_files.extend(glob.glob(os.path.join(os.path.join(config.SPLIT_DIR, source), "*.pgn")))
    
    print(f"\nBắt đầu parse và ghi vào {len(pgn_files)} file PGN...")
    

    for file_path in tqdm(pgn_files, desc="Đang parse file PGN", unit="file"):
        file_name = os.path.basename(file_path)
        # print(f"  -> Đang parse file {i+1}/{len(pgn_files)}: {file_name}")
        
        # Dùng list để gom data của 1 file PGN (giữ RAM thấp)
        chunk_X, chunk_P, chunk_V = [], [], []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            with tqdm(desc=f"Counting {os.path.basename(file_path)}", unit=" games") as pbar:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        pbar.update(1)
                        game_result = game.headers.get('Result', '1/2-1/2')
                        if game_result == '1-0':
                            game_value = 1  # trắng thắng
                        elif game_result == '0-1':
                            game_value = -1 # đen thắng

                        y_value = game_value
                        board = game.board()
                        
                        for move in game.mainline_moves():
                            X_numpy = np.concatenate([
                                utils.board_to_numpy(board), 
                                utils.get_meta_planes_numpy(board)
                            ], axis=0)
                            
                            P_label = utils.move_to_index(move)
                            
                            chunk_X.append(X_numpy)
                            chunk_P.append(P_label)
                            chunk_V.append(y_value)
                            
                            # Thực hiện nước đi
                            board.push(move)
                            
                    except Exception as e:
                        pass # Bỏ qua ván lỗi

        # --- 4. GHI CHUNK NÀY VÀO MMAP ---
        chunk_len = len(chunk_X)
        if chunk_len > 0:
            print(f"     -> Ghi {chunk_len} mẫu vào file 'total'...")
            
            # Kiểm tra xem có bị lố không
            if current_index + chunk_len > total_positions:
                print("!!! LỖI: Số lượng đếm bị sai! Dữ liệu parse ra nhiều hơn dự kiến. Hủy...")
                # Xóa bớt phần bị lố
                chunk_len = total_positions - current_index
                if chunk_len <= 0:
                    break
            
            # Convert list sang numpy array
            X_arr = np.array(chunk_X[:chunk_len], dtype=X_DTYPE)
            P_arr = np.array(chunk_P[:chunk_len], dtype=P_DTYPE)
            V_arr = np.array(chunk_V[:chunk_len], dtype=V_DTYPE)
            
            # Ghi vào file mmap (siêu nhanh)
            total_X_mmap[current_index : current_index + chunk_len] = X_arr
            total_P_mmap[current_index : current_index + chunk_len] = P_arr
            total_V_mmap[current_index : current_index + chunk_len] = V_arr
            
            current_index += chunk_len
            
            # Giải phóng RAM
            del chunk_X, chunk_P, chunk_V, X_arr, P_arr, V_arr

    # --- 5. HOÀN TẤT ---
    print("\nĐã ghi xong! Đang lưu (flush) file...")
    total_X_mmap.flush()
    total_P_mmap.flush()
    total_V_mmap.flush()
    
    print("--- PARSE DỮ LIỆU HOÀN TẤT! ---")
    print(f"Đã ghi tổng cộng {current_index} / {total_positions} mẫu.")
    print(f"3 file 'total' đã sẵn sàng tại: {config.PROCESSED_DIR}")

if __name__ == "__main__":
    """Chuẩn hóa dữ liệu đã sàng lọc"""
    print("Bắt đầu chuẩn hóa..")

    # total_positions = count_total_positions()
    parse_and_write(47432963)

    print("--- Xử lý hoàn tất! ---")
