let playersData = [];
let performanceChart = null;

// Load players when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadPlayers();
    
    // Set up event listeners
    document.getElementById('playerSearch').addEventListener('input', filterPlayers);
    document.getElementById('playerSelect').addEventListener('change', enablePredictButton);
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
});

// Load players from API
async function loadPlayers() {
    try {
        const response = await fetch('/players');
        playersData = await response.json();
        
        if (response.ok) {
            displayPlayers(playersData);
        } else {
            showError('Failed to load players');
        }
    } catch (error) {
        showError('Error loading players: ' + error.message);
    }
}

// Display players in select list
function displayPlayers(players) {
    const select = document.getElementById('playerSelect');
    select.innerHTML = '';
    
    players.forEach(player => {
        const option = document.createElement('option');
        option.value = player.PLAYER_ID;
        option.textContent = `${player.PLAYER_NAME} (${player.GAME_COUNT} games)`;
        select.appendChild(option);
    });
}

// Filter players based on search
function filterPlayers() {
    const searchTerm = document.getElementById('playerSearch').value.toLowerCase();
    const filteredPlayers = playersData.filter(player => 
        player.PLAYER_NAME.toLowerCase().includes(searchTerm)
    );
    displayPlayers(filteredPlayers);
}

// Enable predict button when player is selected
function enablePredictButton() {
    const select = document.getElementById('playerSelect');
    const button = document.getElementById('predictBtn');
    button.disabled = select.value === '';
}

// Make prediction
async function makePrediction() {
    const playerId = document.getElementById('playerSelect').value;
    
    if (!playerId) {
        showError('Please select a player');
        return;
    }
    
    // Show loading state
    const button = document.getElementById('predictBtn');
    button.disabled = true;
    button.textContent = 'Making Prediction...';
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ player_id: playerId })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayPrediction(data);
            loadPlayerHistory(playerId);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Error making prediction: ' + error.message);
    } finally {
        button.disabled = false;
        button.textContent = 'Make Prediction';
    }
}

// Display prediction results
function displayPrediction(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update values
    document.getElementById('playerName').textContent = data.player_name;
    document.getElementById('predictedPoints').textContent = data.predicted_points.toFixed(1);
    document.getElementById('recentAverage').textContent = data.recent_average.toFixed(1);
    
    // Update difference with color
    const diffElement = document.getElementById('difference');
    const diff = data.difference;
    diffElement.textContent = (diff > 0 ? '+' : '') + diff.toFixed(1);
    diffElement.className = diff > 0 ? 'text-success' : 'text-danger';
    
    // Update confidence with color
    const confElement = document.getElementById('confidence');
    confElement.textContent = data.confidence;
    confElement.className = data.confidence === 'High' ? 'text-success' : 'text-warning';
    
    // Hide error
    hideError();
}

// Load player history
async function loadPlayerHistory(playerId) {
    try {
        const response = await fetch(`/player/${playerId}/history`);
        const history = await response.json();
        
        if (response.ok) {
            displayHistory(history);
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Display performance history chart
function displayHistory(history) {
    document.getElementById('historySection').style.display = 'block';
    
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    // Create new chart
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.map((_, i) => `Game ${i + 1}`),
            datasets: [{
                label: 'Points',
                data: history.map(h => h.points),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1
            }, {
                label: 'Rebounds',
                data: history.map(h => h.rebounds),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.1
            }, {
                label: 'Assists',
                data: history.map(h => h.assists),
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Last 20 Games Performance'
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Stats'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Games'
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
}

// Show error message
function showError(message) {
    const alert = document.getElementById('errorAlert');
    const messageSpan = document.getElementById('errorMessage');
    messageSpan.textContent = message;
    alert.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    document.getElementById('errorAlert').style.display = 'none';
}

// Add loading animation to button
function showLoading(button) {
    button.innerHTML = '<span class="loading"></span> Loading...';
    button.disabled = true;
}

// Remove loading animation from button
function hideLoading(button, text) {
    button.innerHTML = text;
    button.disabled = false;
}

// Format numbers for display
function formatNumber(num) {
    return Number(num).toFixed(1);
}

// Add keyboard navigation for player select
document.addEventListener('keydown', function(e) {
    const select = document.getElementById('playerSelect');
    if (select === document.activeElement) {
        if (e.key === 'Enter' && select.value) {
            makePrediction();
        }
    }
});