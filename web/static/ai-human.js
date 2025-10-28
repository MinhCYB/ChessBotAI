const PIECE_PATH = 'img/chesspieces/wikipedia/{piece}.png';
const THEMES = [
    { id: 'theme-wood', label: 'Caramel Choco' },
    { id: 'theme-green', label: 'Mint Dream' },
    { id: 'theme-pink',  label: 'Baby Blush' },
    { id: 'theme-blue',  label: 'Sky Pop' }
];

let selectedTheme = null;
let boardObserver = null;

// Timers & captures state
let whiteTimeSec = 10 * 60; // default 10:00
let blackTimeSec = 10 * 60;
let timerInterval = null;
let activeClock = null; // 'w' or 'b'

let whiteCaptured = [];
let blackCaptured = [];

// ========== Điều hướng với nút Back/Forward ==========
function updateState(view) {
    const current = history.state?.view;
    if (current !== view) {
    history.pushState({ view }, '', `#${view}`);
    } else {
    history.replaceState({ view }, '', `#${view}`);
    }
}

const _showSelector = showSelector;
const _hideSelector = hideSelector;

showSelector = function() {
    _showSelector();
    updateState('select');
};

hideSelector = function() {
    _hideSelector();
    updateState('play');
};

window.addEventListener('popstate', (e) => {
    const view = e.state?.view;
    if (view === 'select') _showSelector();
    else if (view === 'play') _hideSelector();
    else _showSelector();
});


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
    const title = document.createElement('div'); title.className = 'theme-title'; title.innerText = t.label;
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
    document.querySelectorAll('.theme-card').forEach(x => x.classList.remove('selected'));
    if (selectedTheme) {
    const card = document.querySelector(`.theme-card[data-theme="${selectedTheme}"]`);
    if (card) card.classList.add('selected');
    }
}

function hideSelector() {
    document.getElementById('theme-selector').style.display = 'none';
    document.querySelector('h2').style.display = 'block';
    document.getElementById('controls').style.display = 'block';
    document.getElementById('board').style.display = 'block';
    document.getElementById('changeThemeBtn').style.display = 'inline-block';
}

function updateFEN() {
    const fen = game.fen();
    document.getElementById("fenOutput").value = fen;
}

// ===== game logic =====
let game;
let board;
let playerColor = null;

// Helper: format seconds to mm:ss
function fmtTime(s) {
    const mm = Math.floor(s/60).toString().padStart(2,'0');
    const ss = (s%60).toString().padStart(2,'0');
    return `${mm}:${ss}`;
}

function startClockFor(turn) {
    // turn: 'w' or 'b'
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
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; activeClock = null; applyActiveTurnUI(); }
}

function switchClock() {
    if (!activeClock) return;
    activeClock = (activeClock === 'w') ? 'b' : 'w';
    applyActiveTurnUI();
}

function onTimeOut(side) {
    stopClock();
    alert((side === 'w' ? 'White' : 'Black') + ' hết giờ. Kết thúc ván đấu.');
    // optionally end game
}

// helper: lấy DOM top / bottom
function getTopBottomEls() {
    const top = document.getElementById('player-top');
    const bottom = document.getElementById('player-bottom');
    return { top, bottom };
}

function applyActiveTurnUI() {
    const top = document.getElementById('player-top');
    const bottom = document.getElementById('player-bottom');
    top.classList.remove('active-turn');
    bottom.classList.remove('active-turn');

    // xác định ai là trắng / đen theo playerColor
    const humanColor = playerColor; // 'white' hoặc 'black'
    const botColor = (humanColor === 'white') ? 'black' : 'white';

    const whiteEl = (humanColor === 'white') ? bottom : top;
    const blackEl = (humanColor === 'white') ? top : bottom;

    if (activeClock === 'w') whiteEl.classList.add('active-turn');
    if (activeClock === 'b') blackEl.classList.add('active-turn');
}


function renderTimers() {
    const whiteEl = (playerColor === 'white') 
        ? document.querySelector('#player-bottom .player-timer')
        : document.querySelector('#player-top .player-timer');
    const blackEl = (playerColor === 'white') 
        ? document.querySelector('#player-top .player-timer')
        : document.querySelector('#player-bottom .player-timer');

    if (whiteEl) whiteEl.innerText = fmtTime(whiteTimeSec);
    if (blackEl) blackEl.innerText = fmtTime(blackTimeSec);
}


function updateCapturesUI() {
    const whiteEl = (playerColor === 'white') 
        ? document.querySelector('#player-bottom .player-captures')
        : document.querySelector('#player-top .player-captures');
    const blackEl = (playerColor === 'white') 
        ? document.querySelector('#player-top .player-captures')
        : document.querySelector('#player-bottom .player-captures');

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

    if (whiteEl) whiteEl.innerText = whiteCaptured.map(x => pieceToChar(x)).join(' ');
    if (blackEl) blackEl.innerText = blackCaptured.map(x => pieceToChar(x)).join(' ');
}


function handleMoveResult(moveObj) {
    // moveObj — verbose move object from chess.js (has .captured if capture)
    if (!moveObj) return;
    if (moveObj.captured) {
    // If captured piece is lowercase => black piece captured by white, uppercase => white captured by black
    // chess.js returns piece in lowercase always? verbose move has .captured as lowercase letter
    // but also has .color (mover). We'll use moveObj.color (mover's color)
    const moverColor = moveObj.color; // 'w' or 'b'
    const capturedPiece = moveObj.captured; // 'p','n','b','r','q','k'
    if (moverColor === 'w') {
        // white captured a black piece -> add to whiteCaptured
        whiteCaptured.push(capturedPiece.toUpperCase());
    } else {
        blackCaptured.push(capturedPiece.toUpperCase());
    }
    updateCapturesUI();
    }
    // After a legal move, switch clocks to the next side
    switchClock();
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

function onTimeOut(side) {
    stopClock();
    const loser = (side === 'w') ? 'White' : 'Black';
    const winner = (side === 'w') ? 'Black' : 'White';
    alert(`Time out!\n${winner} wins — ${loser} ran out of time.`);
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

    if (!history.state || !history.state.view) {
        history.replaceState({ view: 'selector' }, '', '#select');
    }

    document.getElementById('playWhiteBtn').addEventListener('click', () => {
        if (!selectedTheme) selectedTheme = 'theme-green';
        hideSelector();
        startGame('white');
    });

    document.getElementById('playBlackBtn').addEventListener('click', () => {
        if (!selectedTheme) selectedTheme = 'theme-green';
        hideSelector();
        startGame('black');
    });

    document.getElementById('changeThemeBtn').addEventListener('click', () => {
        showSelector();
        document.getElementById('changeThemeBtn').style.display = 'none';
    });

    document.getElementById("newGameBtn").addEventListener("click", () => {
        board.start();
        updateFEN();
        startGame('white');
    });

    document.getElementById("exportGameBtn").addEventListener("click", () => {
        const pgn = game.pgn();
        const blob = new Blob([pgn], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "game.pgn";
        a.click();
        URL.revokeObjectURL(url);
    });

    document.getElementById('firstBtn').addEventListener('click', () => goToIndex(0));
    document.getElementById('lastBtn').addEventListener('click', () => goToIndex(positionHistory.length - 1));
    document.getElementById('prevBtn').addEventListener('click', () => { if (currentIndex > 0) goToIndex(currentIndex - 1); });
    document.getElementById('nextBtn').addEventListener('click', () => { if (currentIndex < positionHistory.length - 1) goToIndex(currentIndex + 1); });

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


function isValidFenString(s) {
    if (!s || typeof s !== 'string') return false;
    return s.trim().split(' ').length >= 4;
}

function resetClocksAndCaptures() {
    whiteTimeSec = 10 * 60;
    blackTimeSec = 10 * 60;
    whiteCaptured = [];
    blackCaptured = [];
    updateCapturesUI();
    renderTimers();
    stopClock();
}

function startGame(color) {
    const fenInput = document.getElementById('fenInput').value.trim();

    playerColor = color;

    try {
        if (!fenInput || fenInput.toLowerCase() === 'start') game = new Chess(); 
            else if (isValidFenString(fenInput)) game = new Chess(fenInput);
                else game = new Chess(fenInput); 
    } catch (e) {
        alert('❌ Mã FEN không hợp lệ! Vui lòng kiểm tra input FEN hoặc để trống để chơi từ đầu.');
        console.error('Invalid FEN:', fenInput, e);
        return;
    }
    resetClocksAndCaptures();

    const initialPosition = game.fen();
        board = Chessboard('board', {
        draggable: true,
        position: initialPosition,
        orientation: color,
        pieceTheme: PIECE_PATH,
        onDrop: onDrop
    });
    document.getElementById('bottom-controls').style.display = 'block';
    document.getElementById('replay-controls').style.display = 'block';
    document.getElementById('player-top').style.display = 'flex';
    document.getElementById('player-bottom').style.display = 'flex';


    updateFEN();

    if (selectedTheme) {
        applyThemeToBoardContainers(selectedTheme);
        setTimeout(() => applyThemeToBoardContainers(selectedTheme), 50);
    }

    // start clock for side to move
    const toMove = game.turn(); 
    startClockFor(toMove);

    if ((color === 'black' && game.turn() === 'w') || (color === 'white' && game.turn() === 'b')) setTimeout(botMove, 300); 
}

function askPromotion(color) {
    return new Promise(resolve => {
    const dialog = document.getElementById('promotion-dialog');
    dialog.style.display = 'flex';

    // đổi ảnh theo màu
    dialog.querySelectorAll('img').forEach(img => {
        const piece = img.getAttribute('data-piece');
        img.src = `img/chesspieces/wikipedia/${color[0]}${piece.toUpperCase()}.png`;
        img.onclick = () => {
        dialog.style.display = 'none';
        resolve(piece);
        };
    });
    });
}


function onDrop(source, target) {
    if (!playerColor) return 'snapback';

    // Không cho đi sai lượt
    if ((game.turn() === 'w' && playerColor !== 'white') ||
        (game.turn() === 'b' && playerColor !== 'black')) {
        return 'snapback';
    }

    const moves = game.moves({ verbose: true });
    const isPromotion = moves.some(m => m.from === source && m.to === target && m.promotion);

    if (isPromotion) {
        const color = game.turn() === 'w' ? 'white' : 'black';

        const oldPos = game.fen();

        askPromotion(color).then(promotionPiece => {
            const move = game.move({ from: source, to: target, promotion: promotionPiece });
            if (move === null) {
            game.load(oldPos);
            board.position(oldPos); 
            return;
            }

            board.position(game.fen());
            updateFEN();
            handleMoveResult(move);
            handleMoveResult(move);
            
            
            setTimeout(() => {
                if (checkGameEnd()) return;
                setTimeout(botMove, 300);
                
            }, 10);
        

        });

        return 'snapback';
    }

    // ————— NƯỚC ĐI THƯỜNG —————
    const move = game.move({ from: source, to: target, promotion: 'q' });
    if (move === null) return 'snapback';

    // board.position(game.fen());
    updateFEN();
    handleMoveResult(move);
    setTimeout(() => {
        if (checkGameEnd()) return;
        setTimeout(botMove, 300);
        
    }, 10);
}

function botMove() {
    if (checkGameEnd()) return;

    const fen = game.fen();
    fetch('http://127.0.0.1:5000/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ fen: fen })
    })
    .then(res => res.json())
    .then(data => {
    if (data.move) {
        const uci = data.move;
        const from = uci.substring(0, 2);
        const to = uci.substring(2, 4);
        const move = game.move({ from: from, to: to, promotion: 'q' });
        if (move === null) {
        console.error('Bot trả về nước không hợp lệ cho chess.js:', uci);
        return;
        }
        board.position(game.fen());
        updateFEN();

        // use the verbose move object returned by chess.js move call — it contains .captured
        handleMoveResult(move);
        setTimeout(() => {
            checkGameEnd();
        }, 10);
    } else {
        console.error("Bot không trả về nước đi hợp lệ", data);
    }
    })
    .catch(err => console.error('Lỗi gọi bot:', err));
}
