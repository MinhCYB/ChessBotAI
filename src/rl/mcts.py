import torch
import torch.nn as nn
import numpy as np
import math
import chess
from collections import deque
from typing import Dict, Tuple 

from src.utils.utils import move_to_index, index_to_move, board_to_numpy, get_meta_planes_numpy
import config.config as config

class Node:
    """
    Một Nút (Node) trong cây MCTS.
    Đại diện cho một thế cờ.
    """
    def __init__(self, board: chess.Board, prior_p: float, parent=None):
        self.board = board
        self.parent = parent
        self.children: Dict[chess.Move, 'Node'] = {} 
        
        self.visit_count = 0    # N(s,a) - Số lần đi qua
        self.total_value = 0.0  # W(s,a) - Tổng giá trị
        self.q_value = 0.0      # Q(s,a) - Giá trị trung bình (W/N)
        self.prior_p = prior_p  # P(s,a) - Xác suất từ Policy Head

    def select_child(self) -> Tuple[chess.Move, 'Node']:
        """
        Chọn 1 nút con (child) dựa trên công thức PUCT (Q + U).
        """
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        # CPUCT là hằng số khám phá 
        cpuct = config.RL_MCTS_CPUCT 

        # Tính tổng số lượt visit của cha (để tính U)
        parent_visit_sqrt = math.sqrt(self.visit_count)

        for move, child in self.children.items():
            # Công thức PUCT 
            # U = C * P(s,a) * sqrt(N_parent) / (1 + N_child)
            U = cpuct * child.prior_p * parent_visit_sqrt / (1 + child.visit_count)
            
            # Q = Q(s,a) (dưới góc nhìn node con)
            Q = child.q_value
            
            score = Q + U

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child

    def expand_node(self, policy_probs: np.ndarray):
        """
        Mở rộng (expand) nút lá này bằng cách tạo các con
        dựa trên dự đoán của Policy Head.
        """
        for move in self.board.legal_moves:
            move_idx = move_to_index(move)
            # Lấy xác suất P của nước đi này từ model
            prior_p = policy_probs[move_idx]
            
            # Tạo bàn cờ con
            child_board = self.board.copy()
            child_board.push(move)
            
            # Thêm nút con
            self.children[move] = Node(child_board, prior_p, parent=self)

    def backpropagate(self, value: float):
        """
        Cập nhật (backprop) giá trị (value) ngược lên cây
        từ lá về gốc.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            # Giá trị luôn được nhìn từ góc độ của NGƯỜI CHƠI
            # sắp đi nước tại nút CHA. 
            # Vì `value` là kết quả từ nút CON, ta phải lật dấu nó.
            node.total_value += -value 
            node.q_value = node.total_value / node.visit_count
            
            # Lật dấu value cho vòng lặp tiếp theo
            value = -value
            node = node.parent


class MCTS:
    """
    Lớp chính quản lý thuật toán MCTS.
    """
    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device
        self.model.eval() 

    @torch.no_grad()
    def _predict(self, board: chess.Board, history: deque) -> Tuple[np.ndarray, float]:
        """
        Hàm helper: Gọi model neural network.
        Đây là "linh cảm" của MCTS.
        """
        history_stack = np.concatenate(list(history), axis=0)
        meta_planes = get_meta_planes_numpy(board)
        state_tensor_np = np.concatenate([history_stack, meta_planes], axis=0)
        
        state_tensor = torch.from_numpy(
            state_tensor_np.astype(np.float32)
        ).unsqueeze(0).to(self.device)
        
        policy_out, value_out = self.model(state_tensor)
        
        # Xử lý Value
        value = value_out.item() # tensor([[0.123]]) -> 0.123
        
        # Xử lý Policy
        policy_probs = torch.exp(policy_out).squeeze(0).cpu().numpy()
        
        # Lọc ra các nước đi HỢP LỆ (legal moves)
        legal_mask = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=bool)
        legal_indices = [move_to_index(m) for m in board.legal_moves]
        legal_mask[legal_indices] = True
        
        policy_probs = policy_probs * legal_mask
        
        # Chuẩn hóa
        prob_sum = np.sum(policy_probs)
        if prob_sum > 0:
            policy_probs /= prob_sum
        else:
            # Nếu model dự đoán 0% cho tất cả nước hợp lệ
            # thì chia đều xác suất cho các nước hợp lệ
            policy_probs = legal_mask.astype(np.float32) / len(legal_indices)
            
        return policy_probs, value # (policy_probs là 1-D numpy array)

    def search(self, board: chess.Board, history: deque, temperature: float) -> Tuple[chess.Move, np.ndarray]:
        """
        Hàm chính: Chạy N lượt mô phỏng (simulations)
        từ thế cờ (board) hiện tại và chọn ra nước đi.
        """
        
        # Tạo nút gốc
        root_node = Node(board, prior_p=0.0)
        
        # expand gốc
        policy_probs, root_value = self._predict(board, history)
        root_node.expand_node(policy_probs)
        # Backprop giá trị của gốc (để N=1)
        root_node.backpropagate(root_value) 
        
        # Chạy N lượt mô phỏng
        for _ in range(config.RL_MCTS_SIMULATIONS):
            node = root_node
            sim_board = board.copy()
            sim_history = history.copy() 

            # --- Select (Chọn) ---
            # Đi xuống cây, chọn nút con tốt nhất (dùng PUCT)
            # cho đến khi gặp 1 nút lá (chưa có con)
            while node.children:
                move, node = node.select_child()
                sim_board.push(move)
                sim_history.append(board_to_numpy(sim_board)) # Cập nhật history

            # --- b. Expand & Evaluate (Mở rộng & Đánh giá) ---
            # Gặp nút lá.
            value = 0.0
            if sim_board.is_game_over():
                result = sim_board.result()
                if result == '1-0': value = 1.0
                elif result == '0-1': value = -1.0
                else: value = 0.0 # Hòa
            else:
                # Nếu chưa over, dùng model để "đoán"
                policy_probs, value = self._predict(sim_board, sim_history)
                node.expand_node(policy_probs)

            # --- c. Backpropagate (Cập nhật ngược) ---
            # Cập nhật W và N ngược lên gốc
            node.backpropagate(value)
            
        # Trả về "chính sách" (policy) đã được cải thiện
        # (Chính là tỷ lệ visit_count của các con của gốc)
        
        # Y_policy mới 
        mcts_policy_vector = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
        visit_counts = []
        moves = []
        
        for move, child in root_node.children.items():
            move_idx = move_to_index(move)
            mcts_policy_vector[move_idx] = child.visit_count
            visit_counts.append(child.visit_count)
            moves.append(move)
        
        # Chuẩn hóa visit_counts thành policy
        mcts_policy_vector /= np.sum(mcts_policy_vector)
        
        # Chọn nước đi dựa trên "nhiệt độ" (Temperature)
        if temperature == 0:
            # Tham lam (chọn nước visit nhiều nhất)
            best_move_idx = np.argmax(visit_counts)
            chosen_move = moves[best_move_idx]
        else:
            # Phân phối xác suất với "nhiệt độ"
            # (P^(1/T))
            probs = np.array(visit_counts)**(1.0 / temperature)
            probs /= np.sum(probs)
            # Chọn ngẫu nhiên (weighted)
            chosen_move = np.random.choice(moves, p=probs)
            
        return chosen_move, mcts_policy_vector