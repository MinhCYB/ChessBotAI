import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessCNN(nn.Module):
    def __init__(self, num_planes=21, action_size=20480):
        super().__init__()
        """
            Các dữ liệu: 
            - Thế trận của history_length
            - Quyền nhập thành (4 loại)
            - Lượt đi
            - halfmove clock (chuẩn hóa / 50)
            - fullmove number (chuẩn hóa / 100)
            - repetition (1 nếu vị trí lặp, 0 nếu không)
        """
        
        self.num_planes = num_planes
        
        self.conv1 = nn.Conv2d(num_planes, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Kích thước sau khi qua "thân" (256 channel, 8x8)
        self.shared_feature_dim = 256 * 8 * 8

        # --- Policy head (Đầu Policy) ---
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, action_size) # 2*8*8 = 128

        # --- Value head (Đầu Value) --- 
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1) # 1. Conv 1x1 -> 1 channel
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)        # 2. FC (1*8*8=64 -> 256)
        self.value_fc2 = nn.Linear(256, 1)                # 3. FC (256 -> 1) (ra kết quả)

    def forward(self, x):
        # x ban đầu: (batch_size, num_planes, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x bây giờ là feature map chung: (batch_size, 256, 8, 8)

        # --- Policy head  ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 8 * 8) # (batch_size, 128)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1) 

        # --- Value head (Nhánh 2) --- 
        v = F.relu(self.value_bn(self.value_conv(x))) # (batch_size, 1, 8, 8)
        v = v.view(-1, 1 * 8 * 8)                     # (batch_size, 64)
        v = F.relu(self.value_fc1(v))                 # (batch_size, 256)
        v = torch.tanh(self.value_fc2(v))             # (batch_size, 1)
        
        return p, v