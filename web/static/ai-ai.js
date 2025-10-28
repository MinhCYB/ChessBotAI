const PIECE_PATH = 'img/chesspieces/wikipedia/{piece}.png';
const THEMES = [
  { id: 'theme-wood', label: 'Caramel Choco' },
  { id: 'theme-green', label: 'Mint Dream' },
  { id: 'theme-pink',  label: 'Baby Blush' },
  { id: 'theme-blue',  label: 'Sky Pop' }
];

let selectedTheme = null;
let boardObserver = null;

// Timer & capture state
let whiteTimeSec = 10 * 60;
let blackTimeSec = 10 * 60;
let timerInterval = null;
let activeClock = null;

let whiteCaptured = [];
let blackCaptured = [];

let game, board;
let aiRunning = false;

function applyThemeToBoardContainers(themeClass) {
  const boardEl = document.getElementById('board');
  if (!boardEl) return;
  const allEls = [boardEl, ...boardEl.querySelectorAll('*')];
  for (const el of allEls) {
    if (!el.classList) continue;
    Array.from(el.classList).forEach(c => { if (c.startsWith('theme-')) el.classList.remove(c); });
    if (themeClass) el.classList.add(themeClass);
  }
}

function ensureBoardObserver() {
  const boardEl = document.getElementById('board');
  if (!boardEl || boardObserver) return;
  boardObserver = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.addedNodes && m.addedNodes.length > 0 && selectedTheme) {
        setTimeout(() => applyThemeToBoardContainers(selectedTheme), 30);
        break;
      }
    }
  });
  boardObserver.observe(boardEl, { childList: true, subtree: true });
}

// ========== Theme Selector ==========
(function initThemeSelector(){
  const list = document.getElementById('theme-preview-list');
  if (!list) return;

  function fenToObj(fen) {
    const files = ['a','b','c','d','e','f','g','h'];
    const rows = fen.split(' ')[0].split('/');
    let obj = {}; let rank = 8;
    for (let r = 0; r < rows.length; r++) {
      let fileIndex = 0;
      for (const ch of rows[r]) {
        if (/\d/.test(ch)) fileIndex += parseInt(ch,10);
        else {
          const square = files[fileIndex] + rank;
          const piece = (ch === ch.toUpperCase()) ? 'w' + ch.toUpperCase() : 'b' + ch.toUpperCase();
          obj[square] = piece; fileIndex++;
        }
      }
      rank--;
    }
    return obj;
  }

  function pieceToSrc(piece) { return PIECE_PATH.replace('{piece}', piece); }

  function buildMiniBoardWithPieces(themeId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'mini-board ' + themeId;
    const pos = fenToObj('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR');
    for (let r = 8; r >= 1; r--) {
      for (let f = 0; f < 8; f++) {
        const file = ['a','b','c','d','e','f','g','h'][f];
        const square = file + r;
        const isLight = (((8 - r) + f) % 2 === 0);
        const sq = document.createElement('div');
        sq.className = 'mini-square ' + (isLight ? 'white-1e1d7' : 'black-3c85d');
        if (pos[square]) {
          const img = document.createElement('img');
          img.alt = pos[square];
          img.src = pieceToSrc(pos[square]);
          sq.appendChild(img);
        }
        wrapper.appendChild(sq);
      }
    }
    return wrapper;
  }

  THEMES.forEach(t => {
    const card = document.createElement('div');
    card.className = 'theme-card';
    card.setAttribute('data-theme', t.id);
    const mini = buildMiniBoardWithPieces(t.id);
    card.appendChild(mini);
    const title = document.createElement('div');
    title.className = 'theme-title';
    title.innerText = t.label;
    card.appendChild(title);

    card.addEventListener('click', () => {
      selectedTheme = t.id;
      applyThemeToBoardContainers(selectedTheme);
      document.querySelectorAll('.theme-card').forEach(x => x.classList.remove('selected'));
      card.classList.add('selected');
      hideSelector();
    });
    list.appendChild(card);
  });
})();

function showSelector() {
  document.getElementById('theme-selector').style.display = 'block';
  document.querySelector('h2').style.display = 'none';
  document.getElementById('controls').style.display = 'none';
  document.getElementById('board').style.display = 'none';
}

function hideSelector() {
  document.getElementById('theme-selector').style.display = 'none';
  document.querySelector('h2').style.display = 'block';
  document.getElementById('controls').style.display = 'block';
  document.getElementById('board').style.display = 'block';
  document.getElementById('changeThemeBtn').style.display = 'inline-block';
}

// ===== Timer logic =====
function fmtTime(s) {
  const mm = Math.floor(s/60).toString().padStart(2,'0');
  const ss = (s%60).toString().padStart(2,'0');
  return `${mm}:${ss}`;
}

function renderTimers() {
  document.getElementById('white-timer').innerText = fmtTime(whiteTimeSec);
  document.getElementById('black-timer').innerText = fmtTime(blackTimeSec);
}

function startClockFor(turn) {
  activeClock = turn;
  applyActiveTurnUI();
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    if (activeClock === 'w') {
      whiteTimeSec -= 1;
      if (whiteTimeSec <= 0) { whiteTimeSec = 0; onTimeOut('w'); }
    } else {
      blackTimeSec -= 1;
      if (blackTimeSec <= 0) { blackTimeSec = 0; onTimeOut('b'); }
    }
    renderTimers();
  }, 1000);
}

function stopClock() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = null;
  activeClock = null;
  applyActiveTurnUI();
}

function switchClock() {
  if (!activeClock) return;
  activeClock = (activeClock === 'w') ? 'b' : 'w';
  applyActiveTurnUI();
}

function applyActiveTurnUI() {
  const top = document.getElementById('player-top');
  const bottom = document.getElementById('player-bottom');
  top.classList.remove('active-turn');
  bottom.classList.remove('active-turn');
  if (activeClock === 'w') bottom.classList.add('active-turn');
  if (activeClock === 'b') top.classList.add('active-turn');
}

function onTimeOut(side) {
  stopClock();
  const loser = (side === 'w') ? 'AI1' : 'AI2';
  const winner = (side === 'w') ? 'AI2' : 'AI1';
  alert(`Time out!\n${winner} wins — ${loser} ran out of time.`);
  aiRunning = false;
}

// ===== Game logic =====
function updateFEN() {
  const fen = game.fen();
  document.getElementById("fenOutput").value = fen;
}

function resetClocksAndCaptures() {
  whiteTimeSec = 10 * 60;
  blackTimeSec = 10 * 60;
  whiteCaptured = [];
  blackCaptured = [];
  renderTimers();
  stopClock();
}

function handleMoveResult(moveObj) {
  if (!moveObj) return;
  if (moveObj.captured) {
    const moverColor = moveObj.color;
    const capturedPiece = moveObj.captured;
    if (moverColor === 'w') whiteCaptured.push(capturedPiece.toUpperCase());
    else blackCaptured.push(capturedPiece.toUpperCase());
    updateCapturesUI();
  }
  switchClock();
}

function updateCapturesUI() {
    function pieceToChar(piece) {
        if (!piece) return '';
        const p = piece.toUpperCase();
        switch(p) {
            case 'P': return '♟';
            case 'N': return '♞';
            case 'B': return '♝';
            case 'R': return '♜';
            case 'Q': return '♛';
            case 'K': return '♚';
        }
        return '';
    }
    document.getElementById('white-captures').innerText = whiteCaptured.map(x => pieceToChar(x)).join(' ');
    document.getElementById('black-captures').innerText = blackCaptured.map(x => pieceToChar(x)).join(' ');
}

function checkGameEnd() {
    if (game.in_checkmate()) {
        stopClock();
        const loser = game.turn() === 'w' ? 'White' : 'Black';
        const winner = loser === 'White' ? 'Black' : 'White';
        alert(`Checkmate!\n${winner} wins — ${loser} is checkmated.`);
        return true;
    }

    if (game.in_stalemate()) {
        stopClock();
        alert("Draw — Stalemate (no legal moves).");
        return true;
    }

    if (game.in_threefold_repetition()) {
        stopClock();
        alert("Draw — Threefold repetition.");
        return true;
    }

    if (game.insufficient_material()) {
        stopClock();
        alert("Draw — Insufficient material to checkmate.");
        return true;
    }

    if (game.in_draw()) {
        stopClock();
        alert("Draw.");
        return true;
    }

    return false;
}

async function botMove(botId) {
  if (!aiRunning || checkGameEnd()) return;

  const fen = game.fen();
  const res = await fetch("http://127.0.0.1:5000/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen: fen, bot: botId })
  });
  const data = await res.json();

  if (!data.move) {
    console.error("Bot không trả về nước đi hợp lệ", data);
    aiRunning = false;
    return;
  }

  const uci = data.move;
  const from = uci.substring(0, 2);
  const to = uci.substring(2, 4);
  const move = game.move({ from: from, to: to, promotion: 'q' });
  if (!move) {
    console.error('Nước đi không hợp lệ:', uci);
    aiRunning = false;
    return;
  }

  board.position(game.fen());
  updateFEN();
  recordCurrentFen();
  handleMoveResult(move);

  if (checkGameEnd()) return;

  setTimeout(() => {
    const next = game.turn();
    if (next === 'w') botMove('bot1');
    else botMove('bot2');
  }, 500);
}

function startAIGame() {
  aiRunning = true;
  resetClocksAndCaptures();

  const fenInput = document.getElementById('fenInput').value.trim();
  try {
    game = fenInput && fenInput !== 'start' ? new Chess(fenInput) : new Chess();
  } catch {
    alert('❌ Mã FEN không hợp lệ!');
    return;
  }

  board = Chessboard('board', {
    draggable: false,
    position: game.fen(),
    pieceTheme: PIECE_PATH,
    orientation: 'white'
  });

  document.getElementById('bottom-controls').style.display = 'block';
  document.getElementById('replay-controls').style.display = 'block';
  document.getElementById('player-top').style.display = 'flex';
  document.getElementById('player-bottom').style.display = 'flex';

  if (selectedTheme) {
    applyThemeToBoardContainers(selectedTheme);
    setTimeout(() => applyThemeToBoardContainers(selectedTheme), 50);
  }

  startClockFor('w');
  botMove('bot1');
}

let positionHistory = [];
let currentIndex = -1;
let paused = false;

function savePosition(fen) {
  if (currentIndex < positionHistory.length - 1)
    positionHistory = positionHistory.slice(0, currentIndex + 1);

  positionHistory.push(fen);
  if (positionHistory.length > 20) positionHistory.shift(); 
  currentIndex = positionHistory.length - 1;
  updateReplayButtons();
}

function goToIndex(idx) {
  if (idx < 0 || idx >= positionHistory.length) return;
  currentIndex = idx;
  const fen = positionHistory[currentIndex];
  board.position(fen);
  updateFEN();
  recordCurrentFen();
  updateReplayButtons();
}

function recordCurrentFen() {
  const fen = game.fen();
  savePosition(fen);
}

function updateReplayButtons() {
  document.getElementById('firstBtn').disabled = currentIndex <= 0;
  document.getElementById('prevBtn').disabled = currentIndex <= 0;
  document.getElementById('nextBtn').disabled = currentIndex >= positionHistory.length - 1;
  document.getElementById('lastBtn').disabled = currentIndex >= positionHistory.length - 1;
}


window.addEventListener('DOMContentLoaded', () => {
  ensureBoardObserver();
  renderTimers();

  document.getElementById('startAiBtn').addEventListener('click', () => {
    if (!selectedTheme) selectedTheme = 'theme-green';
    hideSelector();
    startAIGame();
  });

  document.getElementById('changeThemeBtn').addEventListener('click', () => {
    showSelector();
    document.getElementById('changeThemeBtn').style.display = 'none';
  });

  document.getElementById('newGameBtn').addEventListener('click', () => {
    aiRunning = false;
    startAIGame();
  });

  document.getElementById('exportGameBtn').addEventListener('click', () => {
    const pgn = game.pgn();
    const blob = new Blob([pgn], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ai_vs_ai_game.pgn";
    a.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById('firstBtn').addEventListener('click', () => goToIndex(0));
  document.getElementById('lastBtn').addEventListener('click', () => goToIndex(positionHistory.length - 1));
  document.getElementById('prevBtn').addEventListener('click', () => {
    if (currentIndex > 0) goToIndex(currentIndex - 1);
  });
  document.getElementById('nextBtn').addEventListener('click', () => {
    if (currentIndex < positionHistory.length - 1) goToIndex(currentIndex + 1);
  });

  document.getElementById('pauseBtn').addEventListener('click', () => {
    paused = !paused;
    aiRunning = !paused;
    const btn = document.getElementById('pauseBtn');
    btn.style.opacity = paused ? '0.6' : '1';

    if (paused) {
        stopClock(); 
    } else {
        const nextTurn = game.turn();
        startClockFor(nextTurn);

        setTimeout(() => {
        if (nextTurn === 'w') botMove('bot1');
        else botMove('bot2');
        }, 500);
    }
  });

});

