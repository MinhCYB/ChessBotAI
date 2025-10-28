import chess
import chess.pgn
import logging
import numpy as np
import os
from tqdm import tqdm 
from collections import deque
import argparse
import csv
import gc

from config.config import *
from src.utils.utils import *

logger = logging.getLogger(__name__)
# --- Hằng số ---\

# Kích thước đầu ra Policy
POLICY_SIZE = 20480
cnt_move = 0
cnt_game = 0

def valid_game(game: chess.pgn.Game, min_elo: int = 0, min_ply = 40):
    """ 
        Kiểm tra ván cờ có đạt chuẩn không
        Đi hơn 20 nước hoàn chỉnh
        Có chiếu hết
        Loại tất cả các ván cờ hòa
    """
    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if white_elo < min_elo or black_elo < min_elo:
            return False 
    except ValueError:
        return False
    
    game_result = game.headers.get('Result', '1/2-1/2')
    if game_result == '1/2-1/2':
        return None
    
    # Lấy bàn cờ ở cuối ván
    # Kiểm tra chiếu hết:
    board = game.end().board() 
    if not board.is_checkmate():
        return False
    
    # ván không được 20 nước hoàn chỉnh thì alex 
    mainline_moves = list(game.mainline_moves())
    if len(mainline_moves) < min_ply:
        return False
    
    return True



def extract_from_pgn(file_path, save_dir, csv_path, file_name, min_elo = 0, min_ply = 0):
    """
        Xử lý một file
    """
    print(f"\nBắt đầu phân tích {os.path.basename(file_path)}")
    sub, trash = 0, 0
    list_valid_game = []
    
    with open(file_path, encoding="utf-8") as pgn:
        with tqdm(desc=f"Parsing {os.path.basename(file_path)}", unit=" games") as pbar:
            while True: 
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception as e:
                    print(f"Lỗi đọc PGN: {e}")
                    break

                if game is None: 
                    break

                if not valid_game(game, min_elo, min_ply):
                    trash += 1
                    continue
                list_valid_game.append(game)

                pbar.update(1)

                if len(list_valid_game) >= 5000: 
                    save(list_valid_game, save_dir, csv_path, file_name, sub, trash)
                    list_valid_game = []
                    gc.collect()
                    trash = 0
                    sub += 1
                
    pbar.close()
    save(list_valid_game, save_dir, csv_path, file_name, sub, trash)

def save(list_game, save_dir, csv_path, file_name, sub=None, trash=0):
    print(f"\nFile {file_name}: Trích xuất được {len(list_game)} ván (đã lọc {trash} ván).")
    name = os.path.splitext(file_name)[0]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{name}_s{sub}", len(list_game)])

    output_path = os.path.join(save_dir, f"{name}_split{sub}.pgn")
    with open(output_path, "w", encoding="utf8") as f: 
        for game in list_game: 
            print(game, file=f)
            print(file=f)

    print("\nLưu thành công!!")

def exists(save_dir, file_name): 
    name = os.path.splitext(file_name)[0]
    output_base_path = os.path.join(save_dir, f"{name}_dataset")
    file_list = [
        f"{output_base_path}.X.npy",
        f"{output_base_path}.y_policy.npy",
        f"{output_base_path}.y_value.npy"
    ]
    for file_name in file_list:
        if not os.path.exists(file_name):
            return False
    return True

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--csvpath', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--minelo', type=int, default=0)
    parser.add_argument('--minply', type=int, default=0)

    args = parser.parse_args()
    file_name = args.filename
    file_path = args.filepath
    csv_path = args.csvpath
    save_dir = args.savedir
    min_elo = args.minelo
    min_ply = args.minply

    extract_from_pgn(file_path, save_dir, csv_path, file_name, min_elo, min_ply)

if __name__ == "__main__":
    main()