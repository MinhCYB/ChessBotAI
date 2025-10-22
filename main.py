import chess 
import chess.pgn
import os
import config.config as config
from tqdm import tqdm
from tqdm import trange
from inference.inference import Inference
from other.minimax2 import MinimaxEngine
from other.stockfish_engine import StockFishEngine
from other.nguyen_engine.engine import ChessEngine

def battle(bot1: Inference, bot2: Inference):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessBot Arena"
    game.headers["White"] = f"{bot1.model_name}"
    game.headers["Black"] = f"{bot2.model_name}"
    game.headers["Result"] = "*"
    node = game
    bots = [bot1, bot2]

    cnt = 0
    while not board.is_game_over(): 
        # move_str = input("input: ")
        # try:
        #     board.push_san(move_str)
        # except Exception as e: 
        #     print("Sai định dạng")
        #     continue

        bot = bots[cnt % 2]
        move, _ = bot.get_action(board)
        board.push(move)
        node = node.add_variation(move)
        turn = "white" if board.turn == chess.WHITE else "black"
        # print(f"{bot.model_name} ({turn}) move {move}")
        # print(board)
        # print("===============================\n")

    result = board.outcome().result()
    info = "0-0"
    if result == '1-0':
        bots[0].cnt_win += 1
        info = f"{bots[0].model_name} win | Tỉ số ({bots[0].cnt_win} / {bots[1].cnt_win})"
    elif result == '0-1':
        bots[1].cnt_win += 1
        info = f"{bots[1].model_name} win | Tỉ số ({bots[0].cnt_win} / {bots[1].cnt_win})"
        
    game.headers['Result'] = result
    with open(config.PGN_PATH, "w", encoding="utf-8") as f:
        print(game, file=f)
    return info
    

def battle_other_engine(bot1: Inference, engine):
    print(f"{bot1.model_name} vs StockFish")
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessBot Arena"
    game.headers["White"] = f"{bot1.model_name}"
    game.headers["Black"] = f"{engine.model_name}"
    game.headers["Result"] = "*"
    node = game
    bots = [bot1, engine]

    cnt = 0
    while not board.is_game_over(): 
        bot = bots[cnt % 2]
        move, _ = bot.get_action(board)
        turn = "white" if board.turn == chess.WHITE else "black"
        print(f"{bot.model_name} ({turn}) move {move}")
        board.push(move)
        node = node.add_variation(move)
        print(board)
        print("===============================\n")
        cnt += 1
    
    print(f"Kết quả: {board.outcome().result()}")
    result = board.outcome().result()
    
    game.headers['Result'] = result
    with open(config.PGN_PATH, "w", encoding="utf-8") as f:
        print(game, file=f)

if __name__ == '__main__': 
    battles = 1000
    bot1 = Inference('sl_model')
    bot2 = Inference('sl_base_model')
    print(f"{bot1.model_name} vs {bot2.model_name}")
    cnt_draw = 0
    for i in trange(battles, unit="game"):
        info = battle(bot1, bot2)
        if info == '0-0':
            cnt_draw += 1
        else: 
            tqdm.write(info)

    print(f"Done:D")
    print(f"{bot1.model_name}: Tỉ lệ thắng={(bot1.cnt_win/battles)*100:.1f}%")
    print(f"{bot2.model_name}: Tỉ lệ thắng={(bot2.cnt_win/battles)*100:.1f}%")
    print(f"Tỉ lệ hòa= {(cnt_draw/battles*100):.1f}%")


    # bot = Inference('sl_base_model')
    # engine = MinimaxEngine(max_depth=3)
    # engine = ChessEngine()
    # battle_other_engine(bot, engine)
    # if isinstance(engine, StockFishEngine):
    #     engine.close()