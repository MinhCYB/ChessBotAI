import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from src.model.architecture import ChessCNN 
from src.preprocessing.dataset import ChessSLDataset
import config.config as config

def train_supervised_model():
    print("--- Bắt đầu Giai đoạn 1: Huấn luyện có giám sát ---")

    # 1. Thiết lập thiết bị 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    writer = SummaryWriter(log_dir="log/sl_training")

    # 2. Tải và chuẩn bị dữ liệu 

    print(f"Quét dữ liệu .npy từ thư mục: {config.PROCESSED_DIR}")
    full_dataset = ChessSLDataset(config.PROCESSED_DIR, config.SOURCES)
    
    print(f"\n---> TỔNG CỘNG: {len(full_dataset)} mẫu dữ liệu bộ file.")
    
    # Chia Train / Validation 
    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Tạo DataLoaders... (Train: {train_size}, Val: {val_size})")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # 3. Xây dựng mô hình
    print(f"Đang xây dựng mô hình ChessCNN...")
    
    model = ChessCNN(
        num_planes=config.TOTAL_PLANES
    ).to(device)

    # model_path = os.path.join(config.CANDIDATE_DIR, f"sl_model.pth")
    # if os.path.exists(model_path):
    #     print(f"Load model in {model_path}")
    #     model.load_state_dict(torch.load(model_path))
        # model.eval()

    # 4. Định nghĩa Loss và Optimizer 
    policy_criterion = nn.NLLLoss().to(device) 
    value_criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 5. Vòng lặp huấn luyện
    best_val_loss = float('inf')
    
    # Đảm bảo thư mục lưu model tồn tại
    os.makedirs(os.path.dirname(config.SL_MODEL_DIR), exist_ok=True)

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        # --- Giai đoạn Huấn luyện (Training) ---
        model.train()
        running_train_loss = 0.0
        running_policy_loss = 0.0
        running_value_loss = 0.0
        
        # Thêm TQDM vào vòng lặp
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
        for X_batch, y_policy_batch, y_value_batch in train_pbar:
            X_batch = X_batch.to(device)
            y_policy_batch = y_policy_batch.to(device)
            y_value_batch = y_value_batch.to(device)
            
            optimizer.zero_grad()
            policy_out, value_out = model(X_batch)
            
            loss_p = policy_criterion(policy_out, y_policy_batch)
            loss_v = value_criterion(value_out, y_value_batch)
            total_loss = loss_p + loss_v # Trọng số 1:1
            
            total_loss.backward()
            optimizer.step()
            
            running_train_loss += total_loss.item()
            running_policy_loss += loss_p.item()
            running_value_loss += loss_v.item() 
            train_pbar.set_postfix(loss=total_loss.item()) # Cập nhật loss lên TQDM

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_policy_loss = running_policy_loss / len(train_loader)
        avg_train_value_loss = running_value_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_policy', avg_train_policy_loss, epoch)
        writer.add_scalar('Loss/train_value', avg_train_value_loss, epoch)

        # --- Giai đoạn Đánh giá (Validation) ---
        model.eval()
        running_val_loss = 0.0
        running_val_policy_loss = 0.0 
        running_val_value_loss = 0.0  
        
        val_pbar = tqdm(val_loader, desc=f"Validate Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for X_batch, y_policy_batch, y_value_batch in val_pbar:
                X_batch = X_batch.to(device)
                y_policy_batch = y_policy_batch.to(device)
                y_value_batch = y_value_batch.to(device)

                policy_out, value_out = model(X_batch)
                
                loss_p = policy_criterion(policy_out, y_policy_batch)
                loss_v = value_criterion(value_out, y_value_batch)
                total_loss = loss_p + loss_v
                
                running_val_loss += total_loss.item()
                running_val_policy_loss += loss_p.item() 
                running_val_value_loss += loss_v.item()
                val_pbar.set_postfix(loss=total_loss.item())

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_policy_loss = running_val_policy_loss / len(val_loader) 
        avg_val_value_loss = running_val_value_loss / len(val_loader)
        print(f"Val Loss:   {avg_val_loss:.4f}")
        
        writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_policy', avg_val_policy_loss, epoch)
        writer.add_scalar('Loss/val_value', avg_val_value_loss, epoch)

        scheduler.step(avg_val_loss)
        
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # 6. Lưu lại model tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path_model = os.path.join(config.SL_MODEL_DIR, f'sl_base_model.pth')
            torch.save(model.state_dict(), path_model)
            print(f"-> New best model saved to {path_model} (Val Loss: {avg_val_loss:.4f})")

    print("--- Huấn luyện Giai đoạn 1 Hoàn tất ---")

if __name__ == "__main__":
    train_supervised_model()