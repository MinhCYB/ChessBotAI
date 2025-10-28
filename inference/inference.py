import torch 
import os
import chess
import numpy as np
import src.utils.utils as utils
import config.config as config
from collections import deque
from src.model.architecture import ChessCNN 
from src.rl.mcts import MCTS
import logging 

logger = logging.getLogger(__name__)


class Inference: 
    def __init__(self, model_name='', history_length=1, use_mcts=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN(num_planes=8+history_length*12).to(self.device)
        self.load(model_name)
        self.cnt_win = 0

        self.use_mcts = use_mcts
        if use_mcts:
            self.mcts_searcher = MCTS(self.model, self.device)

        self.model_name = model_name
        self.history = deque(maxlen=history_length)
        for _ in range(history_length):
            self.history.append(utils.board_to_numpy(chess.Board()))

    def get_action(self, board): 
        if not isinstance(board, chess.Board): 
            # đầu vào là fen
            try: 
                board = chess.Board(board)
            except Exception as e: 
                print(f"Lỗi khi dự đoán nước đi {e}")
                print(f"Chỉ truyền vào chess.Board hoặc mã FEN")
        
        self.history.append(utils.board_to_numpy(board))

        if self.use_mcts: 
            return self.mcts_searcher.search(board, self.history, 0)
        
        history_stack = np.concatenate(list(self.history), axis=0)               # (N*12, 8, 8)
        meta_planes = utils.get_meta_planes_numpy(board)                         # (8, 8, 8)
        state = np.concatenate([history_stack, meta_planes], axis=0)             # (N*12 + 8, 8, 8)

        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_out, value_out = self.model(state_tensor)

        # đánh giá thế cờ
        value = value_out.item()
        policy = policy_out.squeeze(0)

        legal_moves = list(board.legal_moves)
        if not legal_moves: 
            return None, value 
        
        legal_moves = [utils.move_to_index(move) for move in board.legal_moves]
        legal_policy = {}
        for index_move in legal_moves:
            legal_policy[index_move] = policy[index_move]
        
        best_move = utils.index_to_move(max(legal_policy, key=legal_policy.get))
        # logger.info(f"{value}")
        # logger.info(f"\n{legal_policy}")
        
        # print(f"{best_move=}")
        return best_move, value


    def load(self, model_name): 
        model_path = os.path.join(config.CANDIDATE_DIR, f"{model_name}.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()