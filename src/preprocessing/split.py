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

def parser(source):
    data_dir = os.path.join(config.RAW_PGN_DIR, source)
    save_dir = os.path.join(config.SPLIT_DIR, source)
    csv_path = os.path.join(save_dir, f"{source}_info.csv")
    print(f"Bắt đầu quét thư mục: {data_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "num_games"])

    pgn_files = [f for f in os.listdir(data_dir) if f.endswith('.pgn')]
    for file_name in tqdm(pgn_files, desc="Đang đọc file PGN", unit="file"):
        file_path = os.path.join(data_dir, file_name)
        save_dir = os.path.join(config.SPLIT_DIR, source)
        args = [
            r"venv\Scripts\python.exe", "-m", "src.preprocessing.pgn_split",
            "--filename", file_name, 
            "--filepath", file_path,
            "--csvpath", csv_path,
            "--savedir", save_dir,
            "--minelo", ("0" if source == "master" else "2500"),
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
    for source in config.SOURCES:
        parser(source)
    print("--- Xử lý hoàn tất! ---")
