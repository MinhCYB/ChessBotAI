# src/rl/self_play.py

import torch
import torch.nn as nn
import chess
import numpy as np
from collections import deque
import random
import os

from src.rl.mcts import MCTS
from src.utils.utils import * # (Import các hàm helper của bro)
import config.config as config
from src.model.architecture import ChessCNN # Import model

def run_self_play_game(worker_id: int, iter_num: int, model_weights: dict):
    """
    Chạy MỘT ván cờ tự chơi (self-play).
    Hàm này sẽ được gọi song song bởi nhiều "worker".
    
    Nó sẽ trả về list các (state_numpy, mcts_policy_numpy, value).
    (Dạng numpy để tiết kiệm RAM khi gửi qua các process)
    """
    
    # 1. Tải model cho worker này
    device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    
    model = ChessCNN(
        num_planes=config.TOTAL_PLANES 
    ).to(device)
    model.load_state_dict(model_weights)
    model.eval()
    
    mcts_searcher = MCTS(model, device)
    
    # 2. Khởi tạo game
    board = chess.Board()
    
    history = deque(maxlen=config.HISTORY_LENGTH)
    initial_planes = board_to_numpy(board)
    for _ in range(config.HISTORY_LENGTH):
        history.append(initial_planes)
    
    # List để lưu trữ data (X, Y_policy, Y_value)
    # Chúng ta lưu numpy array để tiết kiệm RAM
    game_data_tuples = [] 
    
    current_temp = config.RL_INITIAL_TEMP
    move_count = 0
    
    try:
        # 3. Chơi ván cờ
        while not board.is_game_over():
            # 1. Lấy state_tensor (Input X)
            history_stack = np.concatenate(list(history), axis=0)
            meta_planes = get_meta_planes_numpy(board)
            state_tensor_np = np.concatenate([history_stack, meta_planes], axis=0).astype(np.float32)
            
            # 2. "Suy nghĩ" bằng MCTS
            if move_count >= config.RL_TEMP_DECAY_MOVES:
                current_temp = 0 # (Chuyển sang tham lam)
                
            move, mcts_policy_np = mcts_searcher.search(board, history, current_temp)
            
            # 3. Lưu data (chưa có Value)
            # (state_np, policy_np, value=0.0)
            game_data_tuples.append(
                (state_tensor_np, mcts_policy_np, 0.0)
            )
            
            # 4. Cập nhật game
            board.push(move)
            history.append(board_to_numpy(board)) # (Cập nhật history)
            move_count += 1

        # --- Hết ván cờ ---
        
        # 5. Lấy kết quả
        result = board.result()
        if result == '1-0': final_value = 1.0
        elif result == '0-1': final_value = -1.0
        else: final_value = 0.0 # Hòa

        # 6. Điền (backfill) kết quả Value vào game_data
        final_game_data = []
        
        for i, (state, policy, _) in enumerate(game_data_tuples):
            # Lượt đi của ai thì value là của người đó
            # (Trắng đi nước 0, 2, 4...)
            is_white_move_turn = (i % 2 == 0)
            
            current_value = 0.0
            if (is_white_move_turn and final_value == 1.0) or \
               (not is_white_move_turn and final_value == -1.0):
                current_value = 1.0
            elif (is_white_move_turn and final_value == -1.0) or \
                 (not is_white_move_turn and final_value == 1.0):
                current_value = -1.0
            else:
                current_value = 0.0 # Hòa
                
            final_game_data.append(
                (state, policy, current_value)
            )
        
        # print(f"[Worker {worker_id}] Vòng {iter_num}: Chơi xong {move_count} nước. Kết quả: {result}")
        return final_game_data
        
    except Exception as e:
        print(f"[Worker {worker_id}] LỖI: {e}")
        return [] # Trả về list rỗng nếu có lỗi