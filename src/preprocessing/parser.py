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
import argparse

import config.config as config
import src.utils.utils as utils

def count_total_positions():
    total_positions = 0
    
    # Tìm tất cả file .pgn trong thư mục
    pgn_files = []
    pgn_files.extend(glob.glob(os.path.join(config.SPLIT_DIR, "*.pgn")))
    
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

def save_chunk(chunk_X, chunk_P, chunk_V, chunk_index, dtypes, shard_dir):
    X_DTYPE, P_DTYPE, V_DTYPE = dtypes
    
    print(f"\n---> Đang lưu Chunk {chunk_index:04d} với {len(chunk_X)} mẫu...")
    
    try:
        # 1. Convert list sang numpy array
        X_arr = np.array(chunk_X, dtype=X_DTYPE)
        P_arr = np.array(chunk_P, dtype=P_DTYPE)
        V_arr = np.array(chunk_V, dtype=V_DTYPE)
        
        # 2. Tạo tên file (ví dụ: chunk_0000.X.npy)
        base_name = f"chunk_{chunk_index:04d}"
        path_X = os.path.join(shard_dir, f"{base_name}.X.npy")
        path_P = os.path.join(shard_dir, f"{base_name}.y_policy.npy")
        path_V = os.path.join(shard_dir, f"{base_name}.y_value.npy")

        # 3. Lưu 3 file
        np.save(path_X, X_arr)
        np.save(path_P, P_arr)
        np.save(path_V, V_arr)
        
        print(f"---> Lưu Chunk {chunk_index:04d} thành công.")
        
    except Exception as e:
        print(f"!!! LỖI NGHIÊM TRỌNG khi đang lưu Chunk {chunk_index:04d}: {e}")
    
    finally:
        # Dù lỗi hay không, cũng phải xóa RAM
        chunk_X.clear()
        chunk_P.clear()
        chunk_V.clear()
        if 'X_arr' in locals(): del X_arr, P_arr, V_arr

def parse_and_write_to_shards(total_positions, num_shards=100):
    
    X_ITEM_SHAPE = ((config.HISTORY_LENGTH * 12 + 8), 8, 8)
    X_DTYPE = np.float32
    P_DTYPE = np.int64
    V_DTYPE = np.float32
    DTYPES = (X_DTYPE, P_DTYPE, V_DTYPE)
    
    CHUNK_SIZE = total_positions // num_shards
    if CHUNK_SIZE == 0:
        print("Lỗi: total_positions quá nhỏ so với num_shards!")
        return

    print(f"Tổng số thế cờ: {total_positions}")
    print(f"Số file băm (shard): {num_shards}")
    print(f"Kích thước mỗi chunk: {CHUNK_SIZE} thế cờ")
    
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    print(f"Dữ liệu băm sẽ được lưu tại: {config.PROCESSED_DIR}")
    

    chunk_index = 0
    pgn_files = []
    pgn_files.extend(glob.glob(os.path.join(config.SPLIT_DIR, "*.pgn")))
    
    print(f"\nBắt đầu parse và ghi vào {len(pgn_files)} file PGN...")
    
    chunk_X, chunk_P, chunk_V = [], [], []

    for file_path in tqdm(pgn_files, desc="Đang parse file PGN", unit="file"):
        file_name = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            with tqdm(desc=f"Parsing {file_name}", unit=" games", leave=False) as pbar:
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
                        else:
                            game_value = 0  # Hòa
                        
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
                            
                            board.push(move)

                            if len(chunk_X) >= CHUNK_SIZE:
                                save_chunk(chunk_X, chunk_P, chunk_V, chunk_index, DTYPES, config.PROCESSED_DIR)
                                chunk_index += 1
                        
                    except Exception as e:
                        pass
    if len(chunk_X) > 0:
        print("\nĐang lưu chunk cuối cùng (phần còn sót lại)...")
        save_chunk(chunk_X, chunk_P, chunk_V, chunk_index, DTYPES, config.PROCESSED_DIR)
        chunk_index += 1
    
    print("\n--- PARSE VÀ BĂM HOÀN TẤT! ---")
    print(f"Đã tạo tổng cộng {chunk_index} file băm tại {config.PROCESSED_DIR}.")

if __name__ == "__main__":
    """Chuẩn hóa dữ liệu đã sàng lọc"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--numshards', type=int, default=100, help="Số lượng chunk muốn chia")
    args = parser.parse_args()
    num_shards = args.numshards
    print("Bắt đầu chuẩn hóa..")

    total_positions = count_total_positions()
    parse_and_write_to_shards(total_positions, num_shards=num_shards)

    print("--- Xử lý hoàn tất! ---")
