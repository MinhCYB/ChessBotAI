import chess
import numpy as np

# --- 1. Tạo mapping action <-> index ---
PROMOS = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

def generate_action_mapping():
    """
        Ánh xạ mọi nước đi đơn có thể (from_square, to_square, promotion)
        Tốt được phong [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    """
    actions = []
    for f in range(64):
        for t in range(64):
            for promo in PROMOS:
                actions.append((f, t, promo))

    action_to_idx = {a:i for i,a in enumerate(actions)}
    idx_to_action = {i:a for a,i in action_to_idx.items()}
    return action_to_idx, idx_to_action

ACTION_TO_IDX, IDX_TO_ACTION = generate_action_mapping()
ACTION_SIZE = len(ACTION_TO_IDX)  # 64*64*5 = 20480

# --- 2. Chuyển chess.Move <-> index using mapping ---
def move_to_index(move: chess.Move):
    return ACTION_TO_IDX[(move.from_square, move.to_square, move.promotion)]

def index_to_move(idx: int):
    try:
        f,t,promo = IDX_TO_ACTION[idx]
        return chess.Move(f, t, promotion=promo)
    except:
        return None
    
# --- 3. board -> numpy (C,8,8)
def board_to_numpy(board: chess.Board):
    """"
        Chuyển board sang numpy (12, 8, 8)
    """
    piece_map = board.piece_map()
    state = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_idx = piece_to_index(piece)
        state[piece_idx, row, col] = 1.0
    return state

def get_meta_planes_numpy(board: chess.Board):
    """
        Lấy các thông tin thêm: 
        - Quyền nhập thành
        - Lượt đi
        - halfmove clock (chuẩn hóa / 50)
        - fullmove number (chuẩn hóa / 100)
        - repetition (1 nếu vị trí lặp, 0 nếu không)
    """
    castling_plane = np.zeros((4, 8, 8), dtype=np.float32)
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_plane[0, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_plane[1, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_plane[2, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_plane[3, :, :] = 1.0
        
    turn_plane = np.full((1, 8, 8), dtype=np.float32, fill_value=(1.0 if board.turn == chess.WHITE else 0.0))
    halfmove_plane = np.full((1, 8, 8), dtype=np.float32, fill_value=(board.halfmove_clock / 50.0))
    fullmove_plane = np.full((1, 8, 8), dtype=np.float32, fill_value=(board.fullmove_number / 100.0))
    repetition_plane = np.full((1, 8, 8), dtype=np.float32, fill_value=(1.0 if board.is_repetition() else 0.0))

    return np.concatenate([
        castling_plane, 
        turn_plane,
        halfmove_plane,
        fullmove_plane,
        repetition_plane
    ], axis=0)

def piece_to_index(piece):
    piece_order = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    return piece_order[piece.symbol()]