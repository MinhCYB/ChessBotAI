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

import config.config as config
from src.utils.utils import *

def split():
    data_dir = config.RAW_PGN_DIR
    save_dir = config.SPLIT_DIR
    csv_path = os.path.join(save_dir, f"info.csv")
    print(f"Bắt đầu quét thư mục: {data_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "num_games"])

    pgn_files = [f for f in os.listdir(data_dir) if f.endswith('.pgn')]
    for file_name in tqdm(pgn_files, desc="Đang đọc file PGN", unit="file"):
        file_path = os.path.join(data_dir, file_name)
        save_dir = config.SPLIT_DIR
        args = [
            r"venv\Scripts\python.exe", "-m", "src.preprocessing.pgn_split",
            "--filename", file_name, 
            "--filepath", file_path,
            "--csvpath", csv_path,
            "--savedir", save_dir,
            "--minelo", "2500",
            "--minply", "40"
        ] 
        subprocess.run(args)

    df = pd.read_csv(csv_path)
    total_games = df["num_games"].sum()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Total games", total_games])

if __name__ == "__main__":
    """ Tách và tiền xử lý toàn bộ dữ liệu raw"""
    split()
    print("--- Xử lý hoàn tất! ---")
