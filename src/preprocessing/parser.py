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

import config.config as config
from src.utils.utils import *

def main():
    """ xử lý toàn bộ dữ liệu raw"""
    source = "twic"
    data_dir = os.path.join(config.RAW_PGN_DIR, source)
    print(f"Bắt đầu quét thư mục: {data_dir}")
    
    cnt_game = 0
    cnt_move = 0
    pgn_files = [f for f in os.listdir(data_dir) if f.endswith('.pgn')]
    for file_name in tqdm(pgn_files, desc="Đang đọc file PGN", unit="file"):
        file_path = os.path.join(data_dir, file_name)
        save_dir = os.path.join(config.PROCESSED_DIR, source)
        args = [
            r"venv\Scripts\python.exe", "-m", "src.preprocessing.pgn_parser",
            "--filename", file_name, 
            "--filepath", file_path,
            "--savedir", save_dir,
            "--minelo", "2600"
        ] 
        subprocess.run(args)

        # output = tuple(json.loads(result.stdout))
        # output = result.stdout.strip()
        # output = ast.literal_eval(output)
        # cnt_game += int(output[0])
        # cnt_move += int(output[1])
        # print(print(f"Trích xuất {output[1]} nước đi từ {output[0]} ván (đã lọc).")) 
    
    print("--- Xử lý hoàn tất! ---")
    # print(print(f"Trích xuất {cnt_move} nước đi từ {cnt_game} ván (đã lọc).")) 

if __name__ == "__main__":
    main()