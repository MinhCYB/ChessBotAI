import os
import logging
from pathlib import Path

# config neural
HISTORY_LENGTH = 1
METADATA_LENGTH = 8 # thông tin thêm (4 quyền nhập thành) + (halfmove) + (fullmove) + (repetition) + (lượt đi)
TOTAL_PLANES = HISTORY_LENGTH * 12 + METADATA_LENGTH 
POLICY_OUTPUT_SIZE = 20480

# Siêu tham số huấn luyện 
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
EPOCHS = 20
VAL_SPLIT = 0.2 
NUM_WORKERS = 4

RL_TOTAL_ITERATIONS = 10
RL_NUM_WORKERS = 5
RL_NUM_GAMES_PER_ITER = 100
RL_BUFFER_SIZE_SAMPLES = 800_000
RL_TRAIN_BATCH_SIZE = 512
RL_MCTS_SIMULATIONS = 100
RL_INITIAL_TEMP = 1.0
RL_TEMP_DECAY_MOVES = 35
RL_MCTS_CPUCT = 4.0
RL_TRAIN_EPOCHS = 1 
RL_TRAIN_LR = 0.0001

RL_EVAL_GAMES = 40 
RL_EVAL_WIN_THRESHOLD = 0.55


# Path
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PGN_DIR = DATA_DIR / "raw_pgn"
PROCESSED_DIR = DATA_DIR / "processed"
MERGE_PROCESSED_DIR = DATA_DIR / "merge_processed"
SPLIT_DIR = DATA_DIR / "split"
MODEL_DIR = BASE_DIR / "models"
SL_MODEL_DIR = MODEL_DIR / "sl_base_model"
RL_MODEL_DIR = MODEL_DIR / "rl_best_model"
RL_BEST_MODEL_PATH = RL_MODEL_DIR / "best_model.pth"
CANDIDATE_DIR = MODEL_DIR / "rl_candidate_model"
LOG_PATH = BASE_DIR / "log/debug.log"
PGN_PATH = BASE_DIR / "pgn/battle.pgn"
# SOURCES = ["twic", "master", "tcec", "lichess", "ccrl"]
# SOURCES = ["tcec", "lichess", "ccrl", "master", "twic"]
# SOURCES = ["lichess_2024", "lichess_2025"]  
SOURCES = ["ccrl"]

# --- define log ---
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(funcName)s][%(filename)s:%(lineno)d]: %(message)s",
    datefmt="%H:%M:%S"
)