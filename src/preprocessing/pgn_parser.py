import chess
import chess.pgn
import gc
import logging
import numpy as np
import os
from tqdm import tqdm 
from collections import deque
import argparse
import json

from config.config import *
from src.utils.utils import *

logger = logging.getLogger(__name__)
# --- Hằng số ---\

# Kích thước đầu ra Policy
POLICY_SIZE = 20480
cnt_move = 0
cnt_game = 0

def parse_game(game: chess.pgn.Game, min_elo: int = 0):
    """ 
        Xử lý một ván cờ 
        Trích xuất (state, policy, value) cho mỗi nước đi.
    """
    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))
        if white_elo < min_elo or black_elo < min_elo:
            return [] 
    except ValueError:
        return [] 
    
    game_result = game.headers.get('Result', '1/2-1/2')
    if game_result == '1-0':
        game_value = 1  # trắng thắng
    elif game_result == '0-1':
        game_value = -1 # đen thắng
    else:
        return []
    
    board = game.board()
    data = []
    history = deque(maxlen=HISTORY_LENGTH)
    for _ in range(HISTORY_LENGTH):
        history.append(board_to_numpy(board))

    for move in game.mainline_moves():
        # Biến deque -> list -> stack (N*12, 8, 8)
        history_stack = np.concatenate(list(history), axis=0)               # (N*12, 8, 8)
        meta_planes = get_meta_planes_numpy(board)                          # (8, 8, 8)
        state = np.concatenate([history_stack, meta_planes], axis=0)        # (N*12 + 8, 8, 8)

        actions_index = move_to_index(move)

        # Lấy kết quả từ góc nhìn người chơi hiện tại
        value = game_value if board.turn == chess.WHITE else -game_value
        
        data.append((state, actions_index, value))

        board.push(move)
        history.append(board_to_numpy(board))
    
    return data



def extract_from_pgn(file_path, min_elo = 0):
    """
        Xử lý một file
    """
    print(f"\nBắt đầu phân tích {os.path.basename(file_path)}")
    board_states, action_indices, game_values = [], [], []
    games_parsed = 0

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

                game_data = parse_game(game, min_elo)

                if game_data is None: 
                    continue
                
                pbar.update(1)
                games_parsed += 1
                for state, action_index, value in game_data:
                    board_states.append(state)
                    action_indices.append(action_index)
                    game_values.append(value)
    
    pbar.close()
    print(f"File {os.path.basename(file_path)}: Trích xuất {len(board_states)} nước đi từ {games_parsed} ván (đã lọc).")
    # cnt_move += len(board_states)
    # cnt_game += games_parsed
    return board_states, action_indices, game_values

def save(states, actions, values, save_dir, file_name):
    print(f"Đang nén {file_name} và lưu dữ liệu vào {save_dir}")
    # Chuyển list sang NumPy arrays
    print("Đang chuyển đổi list sang NumPy arrays...")
    X_data = np.array(states, dtype=np.float32)
    y_policy_data = np.array(actions, dtype=np.int64)
    y_value_data = np.array(values, dtype=np.float32)

    name = os.path.splitext(file_name)[0]
    output_base_path = os.path.join(save_dir, f"{name}_dataset")
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    # print(f"  -> Đang lưu {output_base_path}.X.npy")
    X_data = np.array(states, dtype=np.float32)
    np.save(f"{output_base_path}.X.npy", X_data)

    y_policy_data = np.array(actions, dtype=np.int64)
    np.save(f"{output_base_path}.y_policy.npy", y_policy_data)

    y_value_data = np.array(values, dtype=np.float32)
    np.save(f"{output_base_path}.y_value.npy", y_value_data)

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
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--minelo', type=int, default=0)

    args = parser.parse_args()
    file_name = args.filename
    file_path = args.filepath
    save_dir = args.savedir
    min_elo = args.minelo

    if exists(save_dir, file_name):
        return 
    states, actions, values = extract_from_pgn(file_path, min_elo)
    save(states, actions, values, save_dir, file_name)

if __name__ == "__main__":
    main()
    # result = (cnt_game, cnt_move)
    # print(json.dumps(result))