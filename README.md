# ChessBotAI - Huấn luyện Model Cờ Vua

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Method-Supervised%20%2B%20Reinforcement%20Learning-purple" alt="Method">
  <img src="https://img.shields.io/badge/PyTorch-orange" alt="Framework">
  <img src="https://img.shields.io/github/stars/MinhCYB/ChessBotAI?style=social" alt="GitHub Stars">
</p>

Chào mừng đến với `ChessBotAI`! Đây là repository dành cho việc huấn luyện (training) một mô hình AI chơi cờ vua.

Dự án này sử dụng một phương pháp **hybrid (lai)**, kết hợp **Học có Giám sát (Supervised Learning)** để học hỏi từ các ván cờ của con người, sau đó sử dụng **Học Tăng cường (Reinforcement Learning)** thông qua cơ chế tự chơi (self-play) để tinh chỉnh và vượt qua giới hạn của con người.

## Mục Lục

* [Công nghệ Chính](#️-công-nghệ-chính-technology-stack)
* [Chơi với Bot (Giao diện Web)](#-chơi-với-bot-giao-diện-web)
* [Thuật toán](#-thuật-toán)
* [Hướng dẫn huấn luyện mô hình](#-hướng-dẫn-huấn-luyện-mô-hình)
* [Kết quả](#-kết-quả)
  
---
## Công nghệ Chính 

### 1. Lõi AI & Deep Learning (Core AI & Deep Learning)

* **[PyTorch](https://pytorch.org/)**: Framework Deep Learning chính được sử dụng để xây dựng, huấn luyện (train), và thực thi (inference) mô hình `ChessCNN`.
* **[MCTS (Monte Carlo Tree Search)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)**: Thuật toán "bộ não" chính trong Giai đoạn 2 (RL). Nó thực hiện hàng ngàn lượt "mô phỏng" (simulations) để tìm ra phân phối nước đi (policy) tối ưu, thay vì chỉ dựa vào "trực giác" của model.

### 2. Logic Cờ vua (Chess Logic)

* **[python-chess](https://python-chess.readthedocs.io/en/latest/)**: Thư viện cốt lõi để quản lý toàn bộ logic cờ vua. Nó xử lý mọi thứ:
    * Biểu diễn bàn cờ (Board state).
    * Tạo/kiểm tra nước đi (Move generation/validation).
    * Đọc và phân tích file PGN (dữ liệu Giai đoạn 1 - SL).
    * Xử lý FEN (định dạng bàn cờ).

### 3. Hiệu suất & Xử lý Dữ liệu (Performance & Data)

* **[Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)**: Cực kỳ quan trọng trong Giai đoạn 2 (RL). Nó cho phép chạy hàng ngàn ván cờ self-play *song song* (`mp.Pool`), tận dụng tối đa tất cả các nhân CPU để tạo dữ liệu nhanh chóng.
* **[NumPy](https://numpy.org/)**: Dùng để biểu diễn bàn cờ dưới dạng ma trận (bitboards/planes) và là định dạng lưu trữ dữ liệu huấn luyện (`.npy`) hiệu quả.
* **Custom `ChessSLDataset` (Tối ưu I/O cho RAM yếu):** Đây là "vũ khí bí mật" của pipeline dữ liệu. Để xử lý các bộ dữ liệu hàng trăm GB trên máy cá nhân, `Dataset` này sử dụng các kỹ thuật:
    * **Memory Mapping (mmap):** Dùng `np.load(..., mmap_mode='r')` để "ánh xạ" file trên ổ cứng vào bộ nhớ ảo. Nó chỉ tải **chính xác** byte dữ liệu (thế cờ) được yêu cầu vào RAM, giúp RAM sử dụng gần như bằng **0**.
    * **Lazy Indexing & Caching:** Khi khởi tạo, `Dataset` chỉ "quét" (scan) các *header* của file `.npy` để tạo "mục lục" (`cumulative_sizes`). Nó dùng `bisect` (tìm kiếm nhị phân) để tìm ra ngay lập tức một `idx` bất kỳ nằm ở file băm (chunk) nào.
    * **Chunk Shuffling:** Thay vì shuffle 100GB dữ liệu trong RAM (bất khả thi), `Dataset` sẽ **xáo trộn danh sách các file băm** (`random.shuffle(base_names)`) và **chia danh sách file băm** (80% file cho `train`, 20% cho `val`) ngay từ đầu.
* **[ReplayBuffer](...)**: Một cấu trúc dữ liệu (`list`) hoạt động như "bộ nhớ" của AI, lưu trữ hàng triệu nước đi từ các ván self-play gần đây. Lý do chọn `list` là để tăng tốc độ truy xuất ngẫu nhiên.

### 4. 🌐 Ứng dụng & Giám sát (App & Monitoring)

* **[Flask](https://flask.palletsprojects.com/en/3.0.x/)**: Một web framework siêu nhẹ, được dùng để tạo ra giao diện web đơn giản cho phép bạn (và người khác) chơi cờ trực tiếp với model đã huấn luyện.
* **[TensorBoard](https://www.tensorflow.org/tensorboard)**: Được tích hợp (`SummaryWriter`) để theo dõi (monitoring) các chỉ số (loss, win rate, elo) trong quá trình training. Giúp trực quan hóa xem model có đang mạnh lên hay không.
* **[Tqdm](https://github.com/tqdm/tqdm)**: Tạo ra các thanh progress bar (thanh tiến trình) trực quan tiện quan sát tiến độ training.

---

## Chơi với Bot (Giao diện Web)

![](https://github.com/MinhCYB/ChessBotAI/blob/main/mate/ui1.png)

Phần này hướng dẫn bạn cách khởi chạy một web server Flask đơn giản để chơi cờ trực tiếp với model AI đã được huấn luyện.

### Yêu cầu

* Bạn phải có một file model đã huấn luyện (hoặc dùng luôn `sl_lichess_model_v2.pth`) nằm trong thư mục `models/candidate`.
* Hoặc bạn có thể tùy chỉnh lại `bot` trong `web/server`

### Khởi chạy Server

1.  Clone repository:
    ```bash
    git clone "https://github.com/MinhCYB/ChessBotAI.git"
    cd ChessBotAI
    ```
2.  Tạo và kích hoạt môi trường ảo (có thể không cần nhưng khuyến khích để tránh xung đột các phiên bản python trong máy):
    ```bash
    # Linux / macOS / WSL
    python3 -m venv venv
    source venv/bin/activate
    
    # Windows (cmd)
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
    
4.  Sử dụng Flask để chạy ứng dụng web:

    ```bash
    python -m web.server
    ```

5.  Mở trình duyệt của bạn và truy cập: [http://127.0.0.1:5000](http://127.0.0.1:5000). Giờ đây bạn có thể chơi cờ trực tiếp với AI trên giao diện web.

---

## 🔬 Thuật toán

Quy trình huấn luyện của `ChessBotAI` được chia làm hai giai đoạn chính, lấy cảm hứng từ phương pháp của AlphaZero.

### 1. Giai đoạn Học có Giám sát (Supervised Learning - SL)

Đây là giai đoạn "khởi động", mục tiêu là huấn luyện một mô hình "cơ sở" có khả năng bắt chước lối chơi của con người (đại kiện tướng).

* **Dữ liệu:** Sử dụng các file `.npy` (đã xử lý từ PGN) chứa hàng triệu thế cờ.
* **Kiến trúc Model (`ChessCNN`):** Mô hình có hai đầu ra (multi-head):
    *  **Policy Head (Đầu ra Chiến thuật):** Học cách dự đoán *nước đi tiếp theo* mà đại kiện tướng đã chơi.
    *  **Value Head (Đầu ra Giá trị):** Học cách dự đoán *kết quả cuối cùng* của ván cờ (Thắng/Thua/Hòa).
* **Hàm Mất mát (Loss Function):** Tổng của hai hàm loss:
    * **Policy Loss (`NLLLoss`):** Phạt model nếu dự đoán sai nước đi của đại kiện tướng.
    * **Value Loss (`MSELoss`):** Phạt model nếu dự đoán sai kết quả của ván cờ.
* **Kết quả:** Tạo ra một `sl_model.pth` (model cơ sở) có "trực giác" cờ vua cơ bản, sẵn sàng cho giai đoạn 2.

### 2. Giai đoạn Học Tăng cường (Reinforcement Learning - RL)

Đây là giai đoạn "tiến hóa", mục tiêu là để model tự chơi (self-play) và trở nên mạnh mẽ hơn con người. Giai đoạn này là một vòng lặp gồm 3 bước:

#### Bước 1: Tạo Dữ liệu (Self-Play)

* `best_model` (model tốt nhất hiện tại) sẽ tự chơi với chính nó hàng ngàn ván cờ.
* **Dùng Đa luồng (Multiprocessing):** Quá trình này được chạy song song trên nhiều nhân CPU (`mp.Pool`) để tối đa hiệu suất.
* **Dùng MCTS:** Thay vì đi nước ngẫu nhiên, model dùng **MCTS (Monte Carlo Tree Search)** để tìm ra các nước đi "thông minh hơn" và tạo ra một phân phối xác suất (policy) cho các nước đi đó.
* Dữ liệu (`state`, `mcts_policy`, `game_result`) được lưu vào một bộ nhớ đệm gọi là `ReplayBuffer`.

#### Bước 2: Huấn luyện (Training)

* Một `candidate_model` (model ứng cử viên) mới được tạo ra bằng cách copy trọng số từ `best_model`.
* `candidate_model` sau đó được huấn luyện (finetune) bằng cách lấy dữ liệu ngẫu nhiên từ `ReplayBuffer`.
* **Hàm Mất mát (Loss Function) - *Điểm khác biệt chính*:**
    * **Policy Loss (`KLDivLoss`):** Ép model dự đoán *giống hệt* phân phối xác suất mà MCTS đã tìm ra (thay vì chỉ 1 nước đi như SL).
    * **Value Loss (`MSELoss`):** Tương tự SL, học từ kết quả thực tế của ván cờ self-play.

#### Bước 3: Đánh giá (Evaluation)

* `candidate_model` (mới train) sẽ thi đấu với `best_model` (cũ) khoảng 50 ván cờ.
* Trong khi thi đấu, cả hai model đều dùng MCTS ở chế độ "tham lam" (`temperature=0`), tức là luôn chọn nước đi tốt nhất.
* **Quyết định:** Nếu `candidate_model` thắng với tỷ lệ áp đảo (ví dụ: `> 55%`), nó sẽ được "thăng hạng" làm `best_model` mới.
* Vòng lặp quay lại Bước 1, sử dụng `best_model` đã mạnh hơn để tạo dữ liệu.

---

## Chi tiết Hàm Mất Mát (Loss Function)

Đây là phần cốt lõi của thuật toán. Gọi $f_\theta(s)$ là hàm model của chúng ta (với tham số $\theta$), $s$ là một thế cờ (state).
Model trả về 2 giá trị: $f_\theta(s) = (\mathbf{p}, v)$

* $\mathbf{p}$: Một vector **log-probabilities** của nước đi (Policy).
* $v$: Một số vô hướng (scalar) là **giá trị** của thế cờ (Value).

---

### 1. Hàm Loss (SL)

* **Ground Truth (Sự thật):**
    * $z$: Kết quả cuối cùng của ván cờ ($z \in \{-1, 0, 1\}$).
    * $\pi$: Nước đi *duy nhất* mà đại kiện tướng đã chơi.
* **Công thức Tổng:**
    $$L_{SL}(\theta) = L_{value} + L_{policy}$$
* **Chi tiết các thành phần:**
    * **Value Loss (MSELoss):** $L_{value} = (v - z)^2$
        * *Giải thích:* Dùng **MSELoss** để "ép" giá trị $v$ của model cho giống với kết quả $z$ của ván cờ.
    * **Policy Loss (NLLLoss):** $L_{policy} = -\mathbf{p}_{\pi}$
        * *Giải thích:* Dùng **NLLLoss** (Negative Log Likelihood) để "ép" model phải dự đoán ra đúng nước đi $\pi$ mà con người đã chơi (tối đa hóa log-probability của nước đi đó).

---

### 2. Hàm Loss (RL)

* **Ground Truth (Sự thật từ MCTS/Self-play):**
    * $z$: Kết quả cuối cùng của ván cờ *self-play* ($z \in \{-1, 0, 1\}$).
    * $\boldsymbol{\pi}$: Một **phân phối xác suất** (probability distribution) các nước đi do MCTS tìm ra (ví dụ: `e4: 70%, Nf3: 25%`).
* **Công thức Tổng:**
    $$L_{RL}(\theta) = L_{value} + L_{policy}$$
* **Chi tiết các thành phần:**
    * **Value Loss (MSELoss):** $L_{value} = (v - z)^2$
        * *Giải thích:* Tương tự SL, học từ kết quả $z$ của ván cờ self-play.
    * **Policy Loss (Cross-Entropy):** $L_{policy} = -\sum_{a} \boldsymbol{\pi}(a|s) \cdot \mathbf{p}(a|s)$
        * *Giải thích:* Dùng **Cross-Entropy** (trong code PyTorch là `KLDivLoss` với input là log-probs) để "ép" phân phối $\mathbf{p}$ của model cho giống hệt với phân phối $\boldsymbol{\pi}$ "thông minh hơn" của MCTS.

---

### Tóm tắt

| Giai đoạn | Mục tiêu | Policy Loss (L_policy) | Value Loss (L_value) |
| :--- | :--- | :--- | :--- |
| **SL** | Bắt chước 1 nước đi | `NLLLoss` (học 1 index) | `MSELoss` (học $z$ từ data) |
| | | $L_{policy} = -\mathbf{p}_{\pi}$ | $L_{value} = (v - z)^2$ |
| **RL** | Bắt chước 1 phân phối | `KLDivLoss` (học 1 vector) | `MSELoss` (học $z$ từ self-play)|
| | | $L_{policy} = -\sum \boldsymbol{\pi} \cdot \mathbf{p}$ | $L_{value} = (v - z)^2$ |

## Hướng dẫn huấn luyện mô hình

Dưới đây là các bước để huấn luyện model từ đầu.

### Yêu cầu hệ thống

* Python 3.10+
* Git
* (Rất khuyến khích) GPU NVIDIA với CUDA đã được cài đặt. Nếu chưa cài đặt thì lên [torch]("https://pytorch.org/) tìm bản phù hợp với cấu hình máy

### Bước 1: Cài đặt chung

1.  Clone repository:
    ```bash
    git clone "https://github.com/MinhCYB/ChessBotAI.git"
    cd ChessBotAI
    ```
2.  Tạo và kích hoạt môi trường ảo (có thể không cần nhưng khuyến khích để tránh xung đột các phiên bản python trong máy):
    ```bash
    # Linux / macOS / WSL
    python3 -m venv venv
    source venv/bin/activate
    
    # Windows (cmd)
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
---

### Bước 2: Xử lý Dữ liệu (Preprocessing)

Giai đoạn này chuyển đổi các file `.pgn` (dữ liệu thô của con người) thành định dạng `.npy` (hoặc tương tự) mà mô hình có thể "ăn" được.

1.  Đặt tất cả các file PGN của bạn vào thư mục `data/raw_pgn/`.
2.  Chạy script tiền xử lý:

    ```bash
    # Lọc và tách hàng triệu ván cờ từ dữ liệu thô thành những file nhỏ hơn (5000 trận) lưu vào "data/split"
    python -m src.preprocessing.split  
    
    # Đọc dữ liệu từ "data/split" chuyển các file .pgn trong "data/split" thành .npy 
    python -m src.preprocessing.parser --numshards=100 # số lượng chunk data muốn chia
    ```
    Script này sẽ đọc PGN, chuyển đổi mỗi thế cờ thành một vector, gán nhãn (thắng/thua/hòa) và lưu thành các file `.npy` trong `data/processed/`.

---

### Bước 3: Huấn luyện có Giám sát (Supervised Learning)

Sử dụng dữ liệu đã xử lý ở Bước 2 để tạo ra "model cơ sở". Các siêu tham số được lưu ở file `config.py` ở thư mục `config/`

1.  Chạy script huấn luyện ở chế độ `sl`:

    ```bash
    python -m src.train_sl  --input="sl_model"  # Load lại model cũ (nếu có) được lưu tại "models/sl_base_model" mặc định là None
                            --output="sl_model" # Tên model mới lưu tại "models/sl_base_model"
    ```
2.  Quá trình này có thể mất vài giờ (hoặc vài ngày) tùy vào GPU và lượng dữ liệu. Kết quả là một file model (đặt tên theo `--output`) được lưu trong `models/sl_base_model`.

---

### Bước 4: Huấn luyện Tăng cường (Reinforcement Learning)

Sử dụng "model cơ sở" từ Bước 3 để bắt đầu quá trình tự chơi (self-play).

1.  Chạy script huấn luyện ở chế độ `rl`, chỉ định model cơ sở để load:

    ```bash
    python -m src.train_rl --input="sl_model"
    ```
2.  Đây là quá trình chạy rất lâu. Nó sẽ liên tục tự chơi và tự cập nhật model. Model mạnh nhất (`best_model.pth`) sẽ được lưu lại. tại `models/rl_best_model`

---
### Theo dõi quá trình huấn luyện trên tensorboard 

Sử dụng một terminal(hoặc cmd, powershell, `...`) khác:

```bash
python tensorboard --logdir=log
```
Truy cập vào [http://localhost:6006/](http://localhost:6006/) để theo dõi sử thay đổi của chỉ số `loss` của mô hình đang huấn luyện.

---

## Kết quả


### 1. Quá trình Học có Giám sát (SL Training)

Đồ thị dưới đây cho thấy chỉ số `loss` (sai số) của mô hình giảm dần qua các epoch, chứng tỏ mô hình đã học được cách dự đoán kết quả ván đấu từ dữ liệu của con người.

![](https://github.com/MinhCYB/ChessBotAI/blob/main/mate/loss.png)

### 2. Quá trình Học Tăng cường (RL Training)

*cập nhật sau*

### 3. Ví dụ Đánh giá

Model đánh giá một số thế cờ khai cuộc phổ biến:

| Khai cuộc | FEN | Điểm số (Score) |
| :--- | :--- | :---: |
|Sicilian (Najdorf)| `rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3` | `0.16`|
|Sicilian Defense| `rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2` | `0.16`|
|Scandinavian Defense| `rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2` | `0.14`|
|Alekhine’s Defense| `rnbqkb1r/pppppppp/8/8/4P3/5n2/PPPP1PPP/RNBQKBNR w KQkq - 1 2` | `0.06`|
|Bishop’s Opening| `rnbqkbnr/pppp1ppp/8/4p3/2B5/8/PPPPPPPP/RNBQK1NR b KQkq - 2 2` | `-0.05`|
|Ruy Lopez (Spanish Opening)| `rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2` | `-0.09`|
| ... | `...` | `...` |

---

## Cấu trúc Thư mục

```
.
├── config/                 # Chứa file cấu hình siêu tham số
├── data/                   # Chứa dữ liệu
│   ├── processed/          # Dữ liệu .npy đã xử lý cho training (SL)
│   │── split/              # Dữ liệu .pgn thô sau khi chia nhỏ
│   └── raw_pgn/            # Dữ liệu .pgn thô 
├── models/                 # Nơi lưu các file model
│   ├── rl_best_model/      # Lưu model tốt nhất
│   │── rl_candidate_model/ # Lưu ứng cử viên 
│   └── sl_base_model/      # Lưu các model học giám sát
├── src/                    # Mã nguồn chính
│   ├── model/              # Định nghĩa kiến trúc model (Neural Network)
│   ├── preprocessing/      # Tiền xử lý dữ liệu PGN
│   ├── utils/              # Các hàm hỗ trợ chung
│   ├── rl/                 # Các script hỗ trợ training RL
│   ├── train_sl.py         # Logic cho vòng lặp training SL
│   ├── train_rl.py         # Logic cho vòng lặp training RL (self-play)
│   └── ...
├── web/                    # Cài đặt giao diện
│   ├── static/             # html, css, ...
│   └── server.py           
├── .gitignore              # File ignore của Git
├── main.py                 # Để test
├── requirements.txt        # Danh sách các thư viện Python
└── README.md              
```

---

## Liên hệ

MinhCYB - `minhdangquang242006@gmail.com`

Diệu Linh - `ldieu.v2@gmail.com`

Phương Chi - ``


Link dự án: [https://github.com/MinhCYB/ChessBotAI](https://github.com/MinhCYB/ChessBotAI)


