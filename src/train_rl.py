import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from functools import partial
import os
import chess
from collections import deque
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import các file của project
from src.rl.replay_buffer import ReplayBuffer 
from src.rl.self_play import run_self_play_game
from src.rl.mcts import MCTS # <-- Cần MCTS để thi đấu
import config.config as config
from src.model.architecture import ChessCNN
from src.utils.utils import * # <-- Cần utils để chơi cờ

def evaluate_models(best_model: nn.Module, 
                    candidate_model: nn.Module, 
                    device: torch.device) -> float:
    """
    Cho 2 model thi đấu với nhau N ván.
    Trả về tỷ lệ thắng (win rate) của 'candidate_model'.
    """
    print("Bắt đầu thi đấu (Evaluation)...")
    
    # Khởi tạo MCTS cho cả 2 model
    # Khi thi đấu, ta luôn dùng T=0 (tham lam)
    mcts_best = MCTS(best_model, device)
    mcts_candidate = MCTS(candidate_model, device)
    
    candidate_wins = 0
    draws = 0
    
    for i in tqdm(range(config.RL_EVAL_GAMES), desc="Đang thi đấu", leave=False):
        board = chess.Board()
        history = deque(maxlen=config.HISTORY_LENGTH)
        initial_planes = board_to_numpy(board)
        for _ in range(config.HISTORY_LENGTH):
            history.append(initial_planes)
            
        # Lượt 1: Candidate cầm Trắng, Best cầm Đen
        if i % 2 == 0:
            players = {chess.WHITE: mcts_candidate, chess.BLACK: mcts_best}
            player_name = {chess.WHITE: "Candidate", chess.BLACK: "Best"}
        # Lượt 2: Đảo màu
        else:
            players = {chess.WHITE: mcts_best, chess.BLACK: mcts_candidate}
            player_name = {chess.WHITE: "Best", chess.BLACK: "Candidate"}

        while not board.is_game_over():
            # Lấy MCTS của người chơi hiện tại
            current_player_mcts = players[board.turn]
            
            # Lấy nước đi (T=0, tham lam)
            move, _ = current_player_mcts.search(board, history, temperature=0)
            
            board.push(move)
            history.append(board_to_numpy(board))

        # Hết ván, kiểm tra kết quả
        result = board.result()
        if result == "1-0": # Trắng thắng
            if player_name[chess.WHITE] == "Candidate":
                candidate_wins += 1
        elif result == "0-1": # Đen thắng
            if player_name[chess.BLACK] == "Candidate":
                candidate_wins += 1
        else: # Hòa
            draws += 1
            
    win_rate = candidate_wins / config.RL_EVAL_GAMES
    print(f"Thi đấu xong: Candidate thắng {candidate_wins}/{config.RL_EVAL_GAMES} (Hòa: {draws}). Win rate: {win_rate:.2f}")
    return win_rate


if __name__ == "__main__":
    print("--- Bắt đầu Giai đoạn 2: Học tăng cường (RL) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Khởi tạo TensorBoard
    writer = SummaryWriter(log_dir="logs/rl_training")

    # --- SỬA LỖI MODEL LOADING ---
    best_model = ChessCNN(
        num_planes=config.TOTAL_PLANES
    ).to(device)
    
    # 1. Thử load model RL "tốt nhất" (nếu có)
    try:
        best_model.load_state_dict(torch.load(config.RL_BEST_MODEL_PATH))
        print(f"Đã tải model RL tốt nhất từ: {config.RL_BEST_MODEL_PATH}")
    # 2. Nếu không, load model SL (Giai đoạn 1)
    except FileNotFoundError:
        print(f"Không tìm thấy model RL. Tải model SL (base) từ: {config.RL_MODEL_DIR}")
        best_model.load_state_dict(torch.load(config.RL_BEST_MODEL_PATH))
        os.makedirs(os.path.dirname(config.RL_BEST_MODEL_PATH), exist_ok=True)
        torch.save(best_model.state_dict(), config.RL_BEST_MODEL_PATH)
        
    best_model.eval()
    
    # Tạo model ứng cử viên (candidate) - CHỈ ĐỂ GIỮ CHỖ
    # Model thật sẽ được tạo TRONG VÒNG LẶP
    candidate_model = ChessCNN(
        num_planes=config.TOTAL_PLANES
    ).to(device)
    
    replay_buffer = ReplayBuffer(config.RL_BUFFER_SIZE_SAMPLES)
    
    # Lấy model weights (state_dict) ra khỏi GPU, đưa về CPU
    model_weights_cpu = {k: v.cpu() for k, v in best_model.state_dict().items()}

    # --- BẮT ĐẦU CHẠY SONG SONG ---
    mp.set_start_method('spawn', force=True) 
    
    with mp.Pool(config.RL_NUM_WORKERS) as pool:
        for i in range(config.RL_TOTAL_ITERATIONS):
            print(f"\n--- [Vòng lặp RL {i+1}/{config.RL_TOTAL_ITERATIONS}] ---")
            
            # --- Giai đoạn 1: SELF-PLAY (Code của bro đã chuẩn) ---
            print(f"Giai đoạn 1: Đang chạy {config.RL_NUM_GAMES_PER_ITER} ván self-play...")
            best_model.eval() # Đảm bảo model "best" ở chế độ eval
            
            task_func = partial(
                run_self_play_game,
                iter_num=i,
                model_weights=model_weights_cpu
            )
            
            results = [
                pool.apply_async(task_func, (worker_id,)) 
                for worker_id in range(config.RL_NUM_GAMES_PER_ITER)
            ]
            
            total_new_samples = 0
            # Thêm TQDM để xem tiến độ thu hoạch
            for res in tqdm(results, desc="Thu hoạch ván cờ", leave=False):
                game_data = res.get()
                if game_data:
                    replay_buffer.add_game_data(game_data)
                    total_new_samples += len(game_data)
                    
            print(f"Self-play xong. Thu được {total_new_samples} mẫu mới.")
            print(f"Replay Buffer hiện có {len(replay_buffer)} mẫu.")
            writer.add_scalar('SelfPlay/new_samples', total_new_samples, i)
            writer.add_scalar('SelfPlay/buffer_size', len(replay_buffer), i)

            if len(replay_buffer) < config.RL_TRAIN_BATCH_SIZE * 10: # (Ít nhất 10 batch)
                print("Buffer quá nhỏ, tiếp tục self-play...")
                continue

            # --- Giai đoạn 2: TRAINING (Hoàn thiện) ---
            print("Giai đoạn 2: Đang huấn luyện (finetune) model mới...")
            
            # Tải data "lười" từ buffer
            train_loader = DataLoader(
                replay_buffer,
                batch_size=config.RL_TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers=0, 
                pin_memory=True
            )
            
            # Tạo model "ứng cử viên" MỚI và copy weights
            candidate_model.load_state_dict(best_model.state_dict())
            candidate_model.train() # <-- Chuyển sang chế độ train
            
            # --- SỬA LỖI LOSS VÀ OPTIMIZER ---
            # 1. Dùng KLDivLoss cho Policy (vector vs vector)
            policy_criterion = nn.KLDivLoss(reduction='batchmean').to(device)
            # 2. Value loss giữ nguyên
            value_criterion = nn.MSELoss().to(device)
            # 3. Optimizer phải train trên 'candidate_model'
            optimizer = optim.Adam(
                candidate_model.parameters(), 
                lr=config.RL_TRAIN_LR # Dùng LR của RL
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
            # --- HẾT SỬA LỖI ---
            
            running_loss = 0.0
            running_p_loss = 0.0
            running_v_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Training RL Iter {i+1}", leave=False)
            for (X_batch, y_policy_batch, y_value_batch) in train_pbar:
                X_batch = X_batch.to(device)
                y_policy_batch = y_policy_batch.to(device)
                y_value_batch = y_value_batch.to(device)
                
                optimizer.zero_grad()
                policy_out, value_out = candidate_model(X_batch)
                
                loss_p = policy_criterion(policy_out, y_policy_batch)
                loss_v = value_criterion(value_out, y_value_batch)
                total_loss = loss_p + loss_v
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                running_p_loss += loss_p.item()
                running_v_loss += loss_v.item()
                train_pbar.set_postfix(loss=total_loss.item())

            avg_loss = running_loss / len(train_loader)
            print(f"Training xong. Avg Loss: {avg_loss:.4f}")
            writer.add_scalar('Train/total_loss', avg_loss, i)
            writer.add_scalar('Train/policy_loss', running_p_loss / len(train_loader), i)
            writer.add_scalar('Train/value_loss', running_v_loss / len(train_loader), i)

            # --- Giai đoạn 3: EVALUATION (Hoàn thiện) ---
            print("Giai đoạn 3: Đang thi đấu (Evaluate)...")
            best_model.eval()
            candidate_model.eval()
            
            win_rate = evaluate_models(best_model, candidate_model, device)
            writer.add_scalar('Evaluate/candidate_win_rate', win_rate, i)
            
            # CẬP NHẬT MODEL MỚI NHẤT CHO WORKERS (Quan trọng!)
            if win_rate > config.RL_EVAL_WIN_THRESHOLD:
                print(f">>> Candidate THẮNG ({win_rate:.2f})! Nâng cấp 'best_model'.")
                best_model.load_state_dict(candidate_model.state_dict())
                torch.save(best_model.state_dict(), config.RL_BEST_MODEL_PATH)
                
                # Cập nhật weights mới cho vòng self-play tiếp theo
                model_weights_cpu = {k: v.cpu() for k, v in best_model.state_dict().items()}
            else:
                print(f">>> Candidate THUA ({win_rate:.2f}). Giữ lại model cũ.")
                # Không làm gì cả, vòng lặp sau sẽ dùng model_weights_cpu cũ

    print("--- Vòng lặp RL hoàn tất ---")
    writer.close()