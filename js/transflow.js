// ==================== GAME STATE ====================
const gameState = {
    gridSize: 20,
    snake: [{x: 10, y: 10}],
    food: {x: 5, y: 5},
    direction: 'RIGHT',
    gameOver: false,
    score: 0,
    isTraining: false,
    speed: 50 // ms between moves (lower is faster)
};

// ==================== AI FUNCTIONS ====================
async function predictAction() {
    if (!model) return gameState.direction;
    
    const state = getGameStateTensor();
    const actionProbs = model.predict(state);
    const actionValues = await actionProbs.data();
    actionProbs.dispose();
    state.dispose();
    
    const actionIndex = actionValues.indexOf(Math.max(...actionValues));
    return ACTIONS[actionIndex];
}

async function getTrainingAction() {
    const state = getGameStateTensor();
    const actionProbs = model.predict(state);
    const actionValues = await actionProbs.data();
    actionProbs.dispose();
    
    let actionIndex;
    if (Math.random() <= epsilon) {
        actionIndex = Math.floor(Math.random() * ACTIONS.length);
    } else {
        actionIndex = actionValues.indexOf(Math.max(...actionValues));
    }
    
    state.dispose();
    return ACTIONS[actionIndex];
}

// ==================== MODIFIED GAME LOOP ====================
function startGameLoop() {
    if (gameInterval) {
        clearInterval(gameInterval);
    }
    
    gameInterval = setInterval(async () => {
        if (!gameState.gameOver) {
            let action;
            if (gameState.isTraining) {
                action = await getTrainingAction();
            } else if (model) {
                action = await predictAction();
            } else {
                action = gameState.direction;
            }
            
            updateGameState(action);
            renderGame();
            
            if (gameState.gameOver) {
                clearInterval(gameInterval);
                if (gameState.isTraining) {
                    // Jika dalam training, reset game setelah delay
                    await delay(100);
                    resetGame();
                }
            }
        }
    }, gameState.speed);
}

// ==================== MODEL CONFIG ====================
const INPUT_SHAPE = [gameState.gridSize, gameState.gridSize, 1];
const ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT'];
let model;
let trainingStopped = false;
let currentEpisode = 0;
let epsilon = 1.0;
let gameInterval;

// ==================== MODEL CREATION ====================
function createModel() {
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.conv2d({
        inputShape: INPUT_SHAPE,
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    
    // Hidden layers
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.2}));
    
    // Output layer
    model.add(tf.layers.dense({
        units: ACTIONS.length,
        activation: 'softmax'
    }));
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

// ==================== GAME FUNCTIONS ====================
function getGameStateTensor() {
    // Create empty grid
    const grid = Array(gameState.gridSize).fill().map(() => 
        Array(gameState.gridSize).fill(0)
    );
    
    // Mark snake positions (value = -1)
    gameState.snake.forEach(segment => {
        if (segment.x >= 0 && segment.x < gameState.gridSize && 
            segment.y >= 0 && segment.y < gameState.gridSize) {
            grid[segment.y][segment.x] = -1;
        }
    });
    
    // Mark food position (value = 1)
    if (gameState.food.x >= 0 && gameState.food.x < gameState.gridSize && 
        gameState.food.y >= 0 && gameState.food.y < gameState.gridSize) {
        grid[gameState.food.y][gameState.food.x] = 1;
    }
    
    return tf.tensor4d([grid], [1, ...INPUT_SHAPE]);
}

function updateGameState(action) {
    const head = {...gameState.snake[0]};
    
    // Update head position based on action
    switch(action) {
        case 'UP':
            head.y -= 1;
            gameState.direction = 'UP';
            break;
        case 'RIGHT':
            head.x += 1;
            gameState.direction = 'RIGHT';
            break;
        case 'DOWN':
            head.y += 1;
            gameState.direction = 'DOWN';
            break;
        case 'LEFT':
            head.x -= 1;
            gameState.direction = 'LEFT';
            break;
    }
    
    // Check for collisions
    if (head.x < 0 || head.x >= gameState.gridSize || 
        head.y < 0 || head.y >= gameState.gridSize ||
        gameState.snake.some(segment => segment.x === head.x && segment.y === head.y)) {
        gameState.gameOver = true;
        return false;
    }
    
    // Add new head
    gameState.snake.unshift(head);
    
    // Check if food was eaten
    if (head.x === gameState.food.x && head.y === gameState.food.y) {
        gameState.score += 1;
        placeFood();
        return true;
    } else {
        // Remove tail if no food was eaten
        gameState.snake.pop();
        return false;
    }
}

function placeFood() {
    let newFood;
    do {
        newFood = {
            x: Math.floor(Math.random() * gameState.gridSize),
            y: Math.floor(Math.random() * gameState.gridSize)
        };
    } while (gameState.snake.some(segment => 
        segment.x === newFood.x && segment.y === newFood.y));
    
    gameState.food = newFood;
}


// ==================== AI PLAY FUNCTIONS ====================
function toggleAIPlay() {
    if (!model) {
        alert('Model not trained yet! Please train the AI first.');
        return;
    }
    gameState.isAIPlaying = !gameState.isAIPlaying;
    const aiPlayButton = document.getElementById('ai-play');
    
    if (gameState.isAIPlaying) {
        aiPlayButton.textContent = '⏹ Stop AI';
        resetGame();
        startGameLoop();
        gameState.isTraining = false;
        gameState.gameOver = false;
    } else {
        aiPlayButton.textContent = '▶ Let AI Play';
        clearInterval(gameInterval);
        gameInterval = null;
        gameState.gameOver = true;
        gameState.isAIPlaying = false;
        gameState.isTraining = false;
        renderGame();
        document.getElementById('status').textContent = 'AI stopped playing.';
        document.getElementById('start-training').disabled = false;
        document.getElementById('stop-training').disabled = true;
        document.getElementById('ai-play').disabled = false;
    }
}

// ==================== RENDERING FUNCTIONS ====================
function renderGame() {
    const container = document.getElementById('game-container');
    container.innerHTML = '';
    
    // Render snake
    gameState.snake.forEach((segment, index) => {
        const segmentElement = document.createElement('div');
        segmentElement.className = index === 0 ? 'snake-head' : 'snake-segment';
        segmentElement.style.left = `${segment.x * 20}px`;
        segmentElement.style.top = `${segment.y * 20}px`;
        
        // Add slight delay for smooth movement
        segmentElement.style.transitionDelay = `${index * 0.03}s`;
        container.appendChild(segmentElement);
    });
    
    // Render food
    const foodElement = document.createElement('div');
    foodElement.className = 'food';
    foodElement.style.left = `${gameState.food.x * 20}px`;
    foodElement.style.top = `${gameState.food.y * 20}px`;
    container.appendChild(foodElement);
    
    // Update stats
    document.getElementById('score').textContent = gameState.score;
    document.getElementById('length').textContent = gameState.snake.length;
    document.getElementById('episode').textContent = currentEpisode;
    document.getElementById('epsilon').textContent = epsilon.toFixed(2);
}

function resetGame() {
    gameState.snake = [{x: 10, y: 10}];
    gameState.direction = 'RIGHT';
    gameState.gameOver = false;
    gameState.score = 0;
    placeFood();
    renderGame();
    
    // Start the game loop only if not in training mode
    if (!gameState.isTraining) {
        startGameLoop();
    }
}

// ==================== KEYBOARD CONTROLS ====================
function setupKeyboardControls() {
    document.addEventListener('keydown', (e) => {
        if (gameState.isTraining) return; // Disable controls during training
        
        switch(e.key) {
            case 'ArrowUp':
                if (gameState.direction !== 'DOWN') gameState.direction = 'UP';
                break;
            case 'ArrowRight':
                if (gameState.direction !== 'LEFT') gameState.direction = 'RIGHT';
                break;
            case 'ArrowDown':
                if (gameState.direction !== 'UP') gameState.direction = 'DOWN';
                break;
            case 'ArrowLeft':
                if (gameState.direction !== 'RIGHT') gameState.direction = 'LEFT';
                break;
        }
    });
}

// ==================== TRAINING FUNCTIONS ====================
async function trainModel(episodes = 500) {
    gameState.isTraining = true;
    trainingStopped = false;

    const statusElement = document.getElementById('status');
    const startButton = document.getElementById('start-training');
    const stopButton = document.getElementById('stop-training');
    
    startButton.disabled = true;
    stopButton.disabled = false;
    
    // Hyperparameters
    epsilon = 1.0; // Reset exploration rate
    const epsilonMin = 0.01;
    const epsilonDecay = 0.995;
    const gamma = 0.95; // Discount factor
    const batchSize = 32;
    
    // Experience replay memory
    const memory = [];
    const maxMemory = 10000;
    
    // Create model if not exists
    if (!model) {
        model = createModel();
        await displayModelSummary(model);
    }
    
    // Training loop
    for (currentEpisode = 0; currentEpisode < episodes && !trainingStopped; currentEpisode++) {
        resetGame();
        let totalReward = 0;
        let steps = 0;
        
        while (!gameState.gameOver && steps < 500 && !trainingStopped) {
            steps++;
            await delay(10); // Small delay to prevent UI freeze
            
            // Get current state
            const state = getGameStateTensor();
            
            // Predict action probabilities
            const actionProbs = model.predict(state);
            const actionValues = await actionProbs.data();
            actionProbs.dispose();
            
            // Epsilon-greedy action selection
            let actionIndex;
            if (Math.random() <= epsilon) {
                actionIndex = Math.floor(Math.random() * ACTIONS.length);
            } else {
                actionIndex = actionValues.indexOf(Math.max(...actionValues));
            }
            const action = ACTIONS[actionIndex];
            
            // Take action and observe next state and reward
            const ateFood = updateGameState(action);
            const reward = ateFood ? 10 : (gameState.gameOver ? -10 : -0.1);
            totalReward += reward;
            
            const nextState = getGameStateTensor();
            const done = gameState.gameOver;
            
            // Store experience in memory
            memory.push({
                state,
                actionIndex,
                reward,
                nextState,
                done
            });
            
            // If memory is full, remove oldest experience
            if (memory.length > maxMemory) {
                memory.shift();
            }
            
            // Train on batch from memory
            if (memory.length > batchSize) {
                await trainOnBatch(memory, batchSize, gamma);
            }
            
            // Update status periodically
            if (steps % 10 === 0) {
                statusElement.innerHTML = 
                    `<strong>Episode ${currentEpisode + 1}/${episodes}</strong> | ` +
                    `Skor: ${gameState.score} | Langkah: ${steps} | ` +
                    `Total Reward: ${totalReward.toFixed(1)}<br>` +
                    `Epsilon: ${epsilon.toFixed(3)} | Eksplorasi: ${(epsilon * 100).toFixed(1)}%`;
            }
            
            // Render the game with smooth animation
            renderGame();
            await delay(gameState.speed);
        }
        
        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
        
        // Add small delay between episodes
        await delay(100);
    }
    
    // Training finished
    gameState.isTraining = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    
    if (trainingStopped) {
        statusElement.textContent = 'Pelatihan dihentikan oleh pengguna!';
    } else {
        statusElement.textContent = 'Pelatihan selesai! Model telah menyelesaikan semua episode.';
    }
    
    renderGame();
}

async function trainOnBatch(memory, batchSize, gamma) {
    // Sample random batch from memory
    const batch = [];
    for (let i = 0; i < batchSize; i++) {
        const randomIndex = Math.floor(Math.random() * memory.length);
        batch.push(memory[randomIndex]);
    }
    
    // Prepare inputs and targets
    const states = tf.concat(batch.map(item => item.state));
    const nextStates = tf.concat(batch.map(item => item.nextState));
    
    // Predict Q-values for current and next states
    const currentQs = model.predict(states);
    const nextQs = model.predict(nextStates);
    
    // Calculate targets
    const currentQsArray = await currentQs.array();
    const nextQsArray = await nextQs.array();
    
    const targets = [];
    for (let i = 0; i < batch.length; i++) {
        const {actionIndex, reward, done} = batch[i];
        const target = [...currentQsArray[i]];
        
        if (done) {
            target[actionIndex] = reward;
        } else {
            const maxNextQ = Math.max(...nextQsArray[i]);
            target[actionIndex] = reward + gamma * maxNextQ;
        }
        targets.push(target);
    }
    
    const targetTensor = tf.tensor2d(targets);
    
    // Train the model
    await model.fit(states, targetTensor, {
        batchSize: batchSize,
        epochs: 1,
        verbose: 0
    });
    
    // Clean up tensors
    states.dispose();
    nextStates.dispose();
    currentQs.dispose();
    nextQs.dispose();
    targetTensor.dispose();
}

// ==================== UTILITY FUNCTIONS ====================
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function displayModelSummary(model) {
    const summaryContainer = document.getElementById('model-summary');
    summaryContainer.innerHTML = '<strong>Model Summary:</strong><br>';
    
    // Get model summary line by line
    const summaryLines = [];
    model.summary(null, null, (line) => summaryLines.push(line));
    
    // Display summary with formatting
    summaryLines.forEach(line => {
        const lineElement = document.createElement('div');
        lineElement.textContent = line;
        summaryContainer.appendChild(lineElement);
    });
}

function stopTraining() {
    trainingStopped = true;
    gameState.isTraining = false;
    document.getElementById('stop-training').disabled = true;
    document.getElementById('status').textContent += ' (Menghentikan...)';
}

// ==================== INITIALIZATION ====================
function init() {
    resetGame();
    setupKeyboardControls();
    
    // Event listeners
    document.getElementById('start-training').addEventListener('click', async () => {
        document.getElementById('status').textContent = 'Menyiapkan model...';
        await delay(100);
        await trainModel(500);
    });
    
    document.getElementById('stop-training').addEventListener('click', stopTraining);
}

// Start the application
window.onload = init;