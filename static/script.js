// Automatically scroll to the latest message
const chatBox = document.getElementById('chat-box');
chatBox.scrollTop = chatBox.scrollHeight;

// Auto-resize textarea
const textarea = document.querySelector('.chat-input textarea');
textarea.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = `${this.scrollHeight}px`;
});

// Toggle dark mode
const toggleThemeButton = document.getElementById('toggle-theme');

// Check and apply saved theme from localStorage
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
    toggleThemeButton.textContent = '☀️';
}

// Add event listener for theme toggle
toggleThemeButton.addEventListener('click', () => {
    const isDarkMode = document.body.classList.toggle('dark-mode');
    toggleThemeButton.textContent = isDarkMode ? '☀️' : '🌙';
    localStorage.setItem('darkMode', isDarkMode); // Save theme state to localStorage
});

// Initialize charts for each message
document.querySelectorAll('.prob-chart').forEach((canvas, index) => {
    try {
        // Parse probabilities and convert to percentages
        const probabilities = JSON.parse(canvas.dataset.probabilities).map(prob => Math.round(prob * 100)); // Convert to percentage and round

        // Get the 2D context of the canvas
        const ctx = canvas.getContext('2d');

        // Create a new Chart.js bar chart
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'], // Emotion labels
                datasets: [{
                    label: 'Probability (%)', // Label for the dataset
                    data: probabilities, // Use the converted percentages
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(255, 159, 64, 0.2)',
                        'rgba(255, 205, 86, 0.2)',
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(153, 102, 255, 0.2)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(255, 205, 86, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value + '%'; // Add '%' to y-axis labels
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw}%`; // Show percentage in tooltip
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error(`Error initializing chart for canvas ${index}:`, error);
    }
});