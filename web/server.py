from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import importlib
import os
import time
import uuid
import queue
import threading
import chess
from inference.inference import Inference

# BOT_MODULES = {
#     "bot1": "bots.v1_2.predict",
#     "bot2": "bots.bot2.predict",
# }

#### KHỞI TẠO BOT ###
bot = Inference("sl_lichess_model_v2", 1, False)

CALL_ENSURE_READY_AT_START = True
# ==============================================

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# loaded_bots = {}  

# def load_bots():
#     for key, module_path in BOT_MODULES.items():
#         mod = importlib.import_module(module_path)
#         if CALL_ENSURE_READY_AT_START and hasattr(mod, "ensure_ready"):
#             try:
#                 mod.ensure_ready()
#                 print(f"[{key}] ensure_ready() OK")
#             except Exception as e:
#                 print(f"[{key}] ensure_ready() lỗi: {e}")
#         loaded_bots[key] = mod

# load_bots()

# def bot_predict(bot_key: str, fen: str) -> str:
#     """Gọi predict của bot theo key."""
#     mod = loaded_bots.get(bot_key)
#     if mod is None:
#         raise ValueError(f"Bot '{bot_key}' chưa được nạp.")
#     if hasattr(mod, "predict"):
#         return mod.predict(fen)
#     raise ValueError(f"Bot '{bot_key}' không có hàm predict(fen).")

def bot_predict(fen: str) -> str: 
    """ Gọi predict của bot """
    move, _ = bot.get_action(chess.Board(fen))
    return move.uci()

@app.route("/move", methods=["POST"])
def move():
    """
    Body JSON: { "fen": "<FEN>", "bot": "bot1"|"bot2" }
    Return: { "move": "<uci>" }
    """
    data = request.get_json(force=True)
    fen = data.get("fen")
    bot = data.get("bot", "bot1")

    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    try:
        chess.Board(fen=fen)
    except Exception:
        return jsonify({"error": "Invalid FEN"}), 400

    try:
        move_uci = bot_predict(bot, fen)
        return jsonify({"move": move_uci})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

STREAMS = {} 

def sse(event, data):
    return f"event: {event}\ndata: {data}\n\n"

def play_game(stream_id, white_bot, black_bot, start_fen=None, delay=0.5):
    q = STREAMS.get(stream_id)
    if q is None: 
        return

    board = chess.Board(fen=start_fen) if start_fen else chess.Board()
    q.put(sse("fen", board.fen()))

    while not board.is_game_over():
        side_bot = white_bot if board.turn == chess.WHITE else black_bot
        try:
            move_uci = bot_predict(side_bot, board.fen())
        except Exception as e:
            q.put(sse("error", f"{side_bot}: {e}"))
            break

        try:
            mv = chess.Move.from_uci(move_uci)
            if mv not in board.legal_moves:
                q.put(sse("error", f"Nước đi không hợp lệ: {move_uci}"))
                break
            board.push(mv)
        except Exception as e:
            q.put(sse("error", f"Lỗi áp dụng nước đi {move_uci}: {e}"))
            break

        q.put(sse("move", move_uci))
        q.put(sse("fen", board.fen()))
        time.sleep(delay)

    result = (
        "checkmate" if board.is_checkmate() else
        "stalemate" if board.is_stalemate() else
        "insufficient_material" if board.is_insufficient_material() else
        "fivefold_repetition" if board.is_fivefold_repetition() else
        "75_moves_rule" if board.is_seventyfive_moves() else
        "game_over"
    )
    q.put(sse("end", result))
    q.put(None)

@app.route("/start", methods=["POST"])
def start_match():
    """
    Body JSON:
    {
      "white": "bot1", "black": "bot2",
      "fen": "<optional FEN>", "delay": 0.5
    }
    Return: { "stream_id": "<uuid>" }
    """
    data = request.get_json(force=True)
    white = data.get("white", "bot1")
    black = data.get("black", "bot2")
    fen = data.get("fen")
    delay = float(data.get("delay", 0.5))

    # if white not in loaded_bots or black not in loaded_bots:
    #     return jsonify({"error": "Sai tên bot. Dùng 'bot1' hoặc 'bot2'."}), 400

    if fen:
        try:
            chess.Board(fen=fen)
        except Exception:
            return jsonify({"error": "Invalid FEN"}), 400

    sid = str(uuid.uuid4())
    STREAMS[sid] = queue.Queue()
    threading.Thread(
        target=play_game,
        args=(sid, white, black, fen, delay),
        daemon=True
    ).start()
    return jsonify({"stream_id": sid})

@app.route("/stream/<stream_id>")
def stream(stream_id):
    q = STREAMS.get(stream_id)
    if q is None:
        return "Invalid stream", 404

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        STREAMS.pop(stream_id, None)

    return Response(gen(), mimetype="text/event-stream")

@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/ai-vs-ai")
def ai_vs_ai_page():
    return send_from_directory(app.static_folder, "ai-vs-ai.html")

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(debug=True, threaded=True)
