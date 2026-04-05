/**
 * CogniCare: Vanilla JS Implementation
 */
import './index.css';

// --- State Management ---
const appContainer = document.getElementById('app');

// --- Utilities ---
const render = (html) => {
  appContainer.innerHTML = html;

  // Prevent accidental form-submit behavior from dynamically rendered buttons.
  appContainer.querySelectorAll('button:not([type])').forEach((btn) => {
    btn.setAttribute('type', 'button');
  });

  // Re-initialize Lucide icons after each render
  if (window.lucide) {
    window.lucide.createIcons();
  }
};

// Global guard: prevent default browser action for game buttons.
// This avoids accidental page refresh/navigation during gameplay interactions.
document.addEventListener('click', (event) => {
  const btn = event.target.closest('button');
  if (!btn) return;
  event.preventDefault();
}, true);

// --- Dashboard ---
const showDashboard = () => {
  const games = [
    { id: 'memory', title: 'Memory Match', icon: 'grid', color: 'bg-blue-500', desc: 'Find matching pairs of cards' },
    { id: 'sequence', title: 'Sequence Master', icon: 'hash', color: 'bg-indigo-500', desc: 'Repeat the light pattern' },
    { id: 'odd-one', title: 'Odd One Out', icon: 'search', color: 'bg-purple-500', desc: 'Find the different emoji' },
    { id: 'connect', title: 'Connect Dots', icon: 'brain', color: 'bg-orange-500', desc: 'Tap numbers in 1-2-3 order' },
  ];

  const html = `
    <div class="flex-1 flex flex-col fade-in">
      <header class="p-8 pt-12 bg-white border-b">
        <div class="flex items-center gap-3 mb-2">
          <div class="p-2 bg-indigo-600 rounded-xl">
            <i data-lucide="brain" class="w-6 h-6 text-white"></i>
          </div>
          <h1 class="text-2xl font-display font-bold text-slate-900">CogniCare</h1>
        </div>
        <p class="text-slate-500">Daily brain exercises for a healthy mind.</p>
      </header>

      <div class="flex-1 p-6 overflow-y-auto">
        <h2 class="text-lg font-bold text-slate-800 mb-4 px-2">Choose an Exercise</h2>
        <div class="grid gap-4">
          ${games.map(game => `
            <button onclick="window.startGame('${game.id}')" class="flex items-center p-5 bg-white rounded-3xl shadow-sm border border-slate-100 text-left group transition-all hover:shadow-md active:scale-95">
              <div class="w-14 h-14 ${game.color} rounded-2xl flex items-center justify-center mr-5 shadow-inner">
                <i data-lucide="${game.icon}" class="w-7 h-7 text-white"></i>
              </div>
              <div class="flex-1">
                <h3 class="font-display font-bold text-lg text-slate-900">${game.title}</h3>
                <p class="text-sm text-slate-500">${game.desc}</p>
              </div>
            </button>
          `).join('')}
        </div>

        <div class="mt-10 p-6 bg-indigo-50 rounded-3xl border border-indigo-100">
          <h3 class="font-bold text-indigo-900 mb-2">Why play these?</h3>
          <p class="text-sm text-indigo-700 leading-relaxed">
            Regular cognitive stimulation helps maintain neural pathways. These games focus on short-term memory, working memory, visual discrimination, and spatial awareness.
          </p>
        </div>
      </div>
      <footer class="p-4 bg-white border-t text-center">
        <p class="text-[10px] uppercase tracking-widest text-slate-400 font-bold">Brain Health Monitor v1.0</p>
      </footer>
    </div>
  `;
  render(html);
};

// --- Game: Memory Match ---
let memoryState = {
  cards: [],
  flippedIndices: [],
  moves: 0,
  matches: 0
};

const initMemoryGame = () => {
  const icons = ['🍎', '🍌', '🍇', '🍓', '🍒', '🍍', '🥝', '🍉'];
  const deck = [...icons, ...icons]
    .sort(() => Math.random() - 0.5)
    .map((value, index) => ({ id: index, value, isFlipped: false, isMatched: false }));
  
  memoryState = {
    cards: deck,
    flippedIndices: [],
    moves: 0,
    matches: 0
  };
  renderMemoryGame();
};

const handleMemoryClick = (index) => {
  const { cards, flippedIndices } = memoryState;
  if (flippedIndices.length === 2 || cards[index].isFlipped || cards[index].isMatched) return;

  cards[index].isFlipped = true;
  flippedIndices.push(index);
  renderMemoryGame();

  if (flippedIndices.length === 2) {
    memoryState.moves++;
    const [first, second] = flippedIndices;
    if (cards[first].value === cards[second].value) {
      setTimeout(() => {
        cards[first].isMatched = true;
        cards[second].isMatched = true;
        memoryState.flippedIndices = [];
        memoryState.matches++;
        renderMemoryGame();
      }, 500);
    } else {
      setTimeout(() => {
        cards[first].isFlipped = false;
        cards[second].isFlipped = false;
        memoryState.flippedIndices = [];
        renderMemoryGame();
      }, 1000);
    }
  }
};

const renderMemoryGame = () => {
  const html = `
    <div class="flex flex-col h-full slide-in">
      <div class="flex items-center justify-between p-4 bg-white border-b">
        <button onclick="window.showDashboard()" class="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <i data-lucide="arrow-left" class="w-6 h-6"></i>
        </button>
        <h2 class="text-xl font-display font-bold">Memory Match</h2>
        <div class="w-10"></div>
      </div>

      <div class="flex-1 p-4 overflow-y-auto">
        <div class="flex justify-between mb-6 bg-blue-50 p-4 rounded-2xl">
          <div class="text-center">
            <p class="text-xs text-blue-600 font-bold uppercase tracking-wider">Moves</p>
            <p class="text-2xl font-display font-bold text-blue-900">${memoryState.moves}</p>
          </div>
          <div class="text-center">
            <p class="text-xs text-blue-600 font-bold uppercase tracking-wider">Matches</p>
            <p class="text-2xl font-display font-bold text-blue-900">${memoryState.matches}/8</p>
          </div>
        </div>

        <div class="grid grid-cols-4 gap-3">
          ${memoryState.cards.map((card, index) => `
            <div onclick="window.handleMemoryClick(${index})" 
                 class="aspect-square rounded-xl flex items-center justify-center text-3xl cursor-pointer transition-all duration-300 shadow-sm active:scale-95 ${
                   card.isFlipped || card.isMatched 
                     ? 'bg-white border-2 border-blue-200' 
                     : 'bg-blue-500 border-2 border-blue-600'
                 }">
              ${(card.isFlipped || card.isMatched) ? card.value : ''}
            </div>
          `).join('')}
        </div>

        ${memoryState.matches === 8 ? `
          <div class="mt-8 p-6 bg-green-50 rounded-3xl text-center border border-green-100 fade-in">
            <i data-lucide="check-circle-2" class="w-12 h-12 text-green-500 mx-auto mb-2"></i>
            <h3 class="text-xl font-bold text-green-900">Great Job!</h3>
            <p class="text-green-700 mb-4">You found all pairs in ${memoryState.moves} moves.</p>
            <button onclick="window.initMemoryGame()" class="px-6 py-3 bg-green-500 text-white rounded-xl font-bold hover:bg-green-600 transition-colors flex items-center justify-center mx-auto gap-2">
              <i data-lucide="rotate-ccw" class="w-5 h-5"></i> Play Again
            </button>
          </div>
        ` : ''}
      </div>
    </div>
  `;
  render(html);
};

// --- Game: Sequence Master ---
let sequenceState = {
  sequence: [],
  userSequence: [],
  isPlaying: false,
  activeButton: null,
  status: 'idle'
};

const initSequenceGame = () => {
  sequenceState = {
    sequence: [],
    userSequence: [],
    isPlaying: false,
    activeButton: null,
    status: 'idle'
  };
  renderSequenceGame();
};

const startSequenceRound = async () => {
  sequenceState.sequence.push(Math.floor(Math.random() * 4));
  sequenceState.userSequence = [];
  sequenceState.status = 'playing';
  sequenceState.isPlaying = true;
  renderSequenceGame();

  for (let i = 0; i < sequenceState.sequence.length; i++) {
    await new Promise(resolve => setTimeout(resolve, 600));
    sequenceState.activeButton = sequenceState.sequence[i];
    renderSequenceGame();
    await new Promise(resolve => setTimeout(resolve, 400));
    sequenceState.activeButton = null;
    renderSequenceGame();
  }

  sequenceState.isPlaying = false;
  sequenceState.status = 'input';
  renderSequenceGame();
};

const handleSequenceClick = (id) => {
  if (sequenceState.status !== 'input' || sequenceState.isPlaying) return;

  sequenceState.userSequence.push(id);
  
  if (id !== sequenceState.sequence[sequenceState.userSequence.length - 1]) {
    sequenceState.status = 'failed';
    renderSequenceGame();
    return;
  }

  if (sequenceState.userSequence.length === sequenceState.sequence.length) {
    sequenceState.status = 'success';
    renderSequenceGame();
    setTimeout(() => startSequenceRound(), 1000);
  } else {
    renderSequenceGame();
  }
};

const renderSequenceGame = () => {
  const colors = ['bg-red-400', 'bg-blue-400', 'bg-green-400', 'bg-yellow-400'];
  const activeColors = ['bg-red-600', 'bg-blue-600', 'bg-green-600', 'bg-yellow-600'];

  const html = `
    <div class="flex flex-col h-full slide-in">
      <div class="flex items-center justify-between p-4 bg-white border-b">
        <button onclick="window.showDashboard()" class="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <i data-lucide="arrow-left" class="w-6 h-6"></i>
        </button>
        <h2 class="text-xl font-display font-bold">Sequence Master</h2>
        <div class="w-10"></div>
      </div>

      <div class="flex-1 p-6 flex flex-col items-center justify-center">
        <div class="text-center mb-8">
          <p class="text-lg font-medium text-slate-600 mb-2">
            ${sequenceState.status === 'idle' ? 'Watch the pattern and repeat it' : ''}
            ${sequenceState.status === 'playing' ? 'Watching...' : ''}
            ${sequenceState.status === 'input' ? 'Your turn!' : ''}
            ${sequenceState.status === 'failed' ? 'Oops! Try again.' : ''}
            ${sequenceState.status === 'success' ? 'Correct! Next level...' : ''}
          </p>
          <p class="text-4xl font-display font-bold text-slate-900">Level ${sequenceState.sequence.length || 1}</p>
        </div>

        <div class="grid grid-cols-2 gap-4 w-full max-w-xs">
          ${[0, 1, 2, 3].map(id => `
            <button onclick="window.handleSequenceClick(${id})" 
                    class="aspect-square rounded-3xl shadow-lg transition-colors duration-200 active:scale-90 ${
                      sequenceState.activeButton === id ? activeColors[id] : colors[id]
                    }">
            </button>
          `).join('')}
        </div>

        ${sequenceState.status === 'idle' ? `
          <button onclick="window.startSequenceRound()" class="mt-12 px-8 py-4 bg-indigo-600 text-white rounded-2xl font-bold text-lg shadow-xl hover:bg-indigo-700 transition-all active:scale-95">
            Start Game
          </button>
        ` : ''}

        ${sequenceState.status === 'failed' ? `
          <button onclick="window.initSequenceGame()" class="mt-12 px-8 py-4 bg-red-600 text-white rounded-2xl font-bold text-lg shadow-xl hover:bg-red-700 transition-all flex items-center gap-2 active:scale-95">
            <i data-lucide="rotate-ccw" class="w-5 h-5"></i> Try Again
          </button>
        ` : ''}
      </div>
    </div>
  `;
  render(html);
};

// --- Game: Odd One Out ---
let oddState = {
  level: 1,
  grid: [],
  targetIndex: -1,
  gameOver: false
};

const initOddGame = () => {
  oddState.level = 1;
  oddState.gameOver = false;
  generateOddGrid();
};

const generateOddGrid = () => {
  const emojiSets = [
    ['🍎', '🍏'], ['🐶', '🐱'], ['🌞', '🌙'], ['🚗', '🚲'], 
    ['⚽', '🏀'], ['🍔', '🍕'], ['🎸', '🎻'], ['🏠', '🏢']
  ];
  const set = emojiSets[Math.floor(Math.random() * emojiSets.length)];
  const size = oddState.level < 5 ? 4 : oddState.level < 10 ? 9 : 16;
  const newGrid = Array(size).fill(set[0]);
  const target = Math.floor(Math.random() * size);
  newGrid[target] = set[1];
  oddState.grid = newGrid;
  oddState.targetIndex = target;
  renderOddGame();
};

const handleOddClick = (index) => {
  if (oddState.gameOver) return;
  if (index === oddState.targetIndex) {
    oddState.level++;
    generateOddGrid();
  } else {
    oddState.gameOver = true;
    renderOddGame();
  }
};

const renderOddGame = () => {
  const cols = oddState.grid.length === 4 ? 'grid-cols-2' : oddState.grid.length === 9 ? 'grid-cols-3' : 'grid-cols-4';
  const html = `
    <div class="flex flex-col h-full slide-in">
      <div class="flex items-center justify-between p-4 bg-white border-b">
        <button onclick="window.showDashboard()" class="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <i data-lucide="arrow-left" class="w-6 h-6"></i>
        </button>
        <h2 class="text-xl font-display font-bold">Odd One Out</h2>
        <div class="w-10"></div>
      </div>

      <div class="flex-1 p-6 flex flex-col items-center">
        <div class="w-full mb-8 flex justify-between items-center bg-purple-50 p-4 rounded-2xl">
          <div>
            <p class="text-xs text-purple-600 font-bold uppercase">Level</p>
            <p class="text-2xl font-display font-bold text-purple-900">${oddState.level}</p>
          </div>
          <p class="text-sm font-medium text-purple-700">Find the different emoji</p>
        </div>

        <div class="grid ${cols} gap-3 w-full max-w-sm">
          ${oddState.grid.map((emoji, i) => `
            <button onclick="window.handleOddClick(${i})" 
                    class="aspect-square bg-white rounded-2xl shadow-sm border border-slate-100 text-4xl flex items-center justify-center hover:bg-slate-50 transition-colors active:scale-90">
              ${emoji}
            </button>
          `).join('')}
        </div>

        ${oddState.gameOver ? `
          <div class="mt-12 p-8 bg-white rounded-3xl shadow-2xl text-center border-2 border-red-100 fade-in">
            <i data-lucide="alert-circle" class="w-16 h-16 text-red-500 mx-auto mb-4"></i>
            <h3 class="text-2xl font-bold text-slate-900 mb-2">Game Over</h3>
            <p class="text-slate-600 mb-6">You reached Level ${oddState.level}</p>
            <button onclick="window.initOddGame()" class="w-full py-4 bg-purple-600 text-white rounded-2xl font-bold text-lg shadow-lg hover:bg-purple-700 transition-all active:scale-95">
              Try Again
            </button>
          </div>
        ` : ''}
      </div>
    </div>
  `;
  render(html);
};

// --- Game: Connect Dots ---
let connectState = {
  dots: [],
  nextDot: 1,
  completed: false,
  width: 0,
  height: 0
};

const initConnectGame = () => {
  const container = document.getElementById('game-container');
  if (!container) {
    // Initial render to get the container
    renderConnectGameSkeleton();
    setTimeout(initConnectGame, 100);
    return;
  }

  const rect = container.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  const newDots = [];
  const count = 8;
  const padding = 40;
  const minDistance = 60;

  for (let i = 1; i <= count; i++) {
    let x, y, tooClose;
    let attempts = 0;
    do {
      tooClose = false;
      x = padding + Math.random() * (width - padding * 2);
      y = padding + Math.random() * (height - padding * 2);
      for (const dot of newDots) {
        const dist = Math.sqrt(Math.pow(x - dot.x, 2) + Math.pow(y - dot.y, 2));
        if (dist < minDistance) { tooClose = true; break; }
      }
      attempts++;
    } while (tooClose && attempts < 50);
    newDots.push({ id: i, x, y, isConnected: false });
  }

  connectState = {
    dots: newDots,
    nextDot: 1,
    completed: false,
    width,
    height
  };
  renderConnectGame();
};

const handleConnectClick = (id) => {
  if (id === connectState.nextDot) {
    connectState.dots[id - 1].isConnected = true;
    if (id === connectState.dots.length) {
      connectState.completed = true;
    } else {
      connectState.nextDot++;
    }
    renderConnectGame();
  }
};

const renderConnectGameSkeleton = () => {
  const html = `
    <div class="flex flex-col h-full slide-in">
      <div class="flex items-center justify-between p-4 bg-white border-b">
        <button onclick="window.showDashboard()" class="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <i data-lucide="arrow-left" class="w-6 h-6"></i>
        </button>
        <h2 class="text-xl font-display font-bold">Connect Dots</h2>
        <div class="w-10"></div>
      </div>
      <div id="game-container" class="flex-1 relative overflow-hidden bg-slate-50"></div>
    </div>
  `;
  render(html);
};

const renderConnectGame = () => {
  const html = `
    <div class="flex flex-col h-full slide-in">
      <div class="flex items-center justify-between p-4 bg-white border-b">
        <button onclick="window.showDashboard()" class="p-2 hover:bg-slate-100 rounded-full transition-colors">
          <i data-lucide="arrow-left" class="w-6 h-6"></i>
        </button>
        <h2 class="text-xl font-display font-bold">Connect Dots</h2>
        <div class="w-10"></div>
      </div>

      <div id="game-container" class="flex-1 relative overflow-hidden bg-slate-50">
        <div class="absolute top-4 left-4 right-4 bg-orange-50 p-3 rounded-xl border border-orange-100 text-center z-10">
          <p class="text-orange-800 font-medium">Tap the numbers in order: <span class="font-bold text-lg">${connectState.nextDot}</span></p>
        </div>

        <svg class="absolute inset-0 w-full h-full pointer-events-none">
          ${connectState.dots.map((dot, i) => {
            if (i > 0 && connectState.dots[i].isConnected && connectState.dots[i-1].isConnected) {
              return `<line x1="${connectState.dots[i-1].x}" y1="${connectState.dots[i-1].y}" x2="${dot.x}" y2="${dot.y}" stroke="#f97316" stroke-width="4" stroke-linecap="round" />`;
            }
            return '';
          }).join('')}
        </svg>

        ${connectState.dots.map(dot => `
          <button onclick="window.handleConnectClick(${dot.id})" 
                  style="left: ${dot.x - 24}px; top: ${dot.y - 24}px;"
                  class="absolute w-12 h-12 rounded-full flex items-center justify-center font-bold text-xl shadow-md transition-all z-20 active:scale-90 ${
                    dot.isConnected ? 'bg-orange-500 text-white' : 'bg-white text-slate-900 border-2 border-slate-200'
                  }">
            ${dot.id}
          </button>
        `).join('')}

        ${connectState.completed ? `
          <div class="absolute bottom-10 left-6 right-6 p-8 bg-white rounded-3xl shadow-2xl text-center border-2 border-orange-100 z-30 fade-in">
            <h3 class="text-2xl font-bold text-slate-900 mb-2">Excellent!</h3>
            <p class="text-slate-600 mb-6">You connected all the dots perfectly.</p>
            <button onclick="window.initConnectGame()" class="w-full py-4 bg-orange-500 text-white rounded-2xl font-bold text-lg shadow-lg hover:bg-orange-600 transition-all active:scale-95">
              Play Again
            </button>
          </div>
        ` : ''}
      </div>
    </div>
  `;
  render(html);
};

// --- Global API ---
window.showDashboard = showDashboard;
window.startGame = (id) => {
  if (id === 'memory') initMemoryGame();
  if (id === 'sequence') initSequenceGame();
  if (id === 'odd-one') initOddGame();
  if (id === 'connect') initConnectGame();
};
window.handleMemoryClick = handleMemoryClick;
window.initMemoryGame = initMemoryGame;
window.handleSequenceClick = handleSequenceClick;
window.startSequenceRound = startSequenceRound;
window.initSequenceGame = initSequenceGame;
window.handleOddClick = handleOddClick;
window.initOddGame = initOddGame;
window.handleConnectClick = handleConnectClick;
window.initConnectGame = initConnectGame;

// --- Initial Load ---
window.onload = () => {
  showDashboard();
};
