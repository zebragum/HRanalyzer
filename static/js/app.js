document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyzeForm');
    const urlInput = document.getElementById('urlInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const url = urlInput.value.trim();
        if (!url) return;

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        statusMessage.textContent = 'Submitting...';
        resultContainer.innerHTML = '';

        fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        })
            .then(resp => resp.json())
            .then(() => {
                // Start polling progress
                pollProgress();
            })
            .catch(err => {
                console.error(err);
                statusMessage.textContent = 'Error submitting video.';
            });
    });

    function pollProgress() {
        fetch('/progress')
            .then(resp => resp.json())
            .then(data => {
                if (!data) return;
                statusMessage.textContent = data.message || data.status;
                if (typeof data.progress === 'number') {
                    progressBar.style.width = `${data.progress}%`;
                }
                if (data.status === 'completed') {
                    displayResults(data.result);
                } else if (data.status === 'error') {
                    statusMessage.textContent = data.error || 'Unknown error';
                } else {
                    setTimeout(pollProgress, 3000);
                }
            })
            .catch(err => {
                console.error(err);
                setTimeout(pollProgress, 5000);
            });
    }

    function displayResults(result) {
        if (!result) return;
        let html = `<h2>Results</h2>`;
        html += `<p><strong>BPM:</strong> ${result.bpm.toFixed(1)}</p>`;
        html += `<img src="data:image/png;base64,${result.plot_url}" alt="Signal Plot" style="max-width:100%;" />`;
        if (result.magnified_video) {
            const magFile = result.magnified_video.split('/').pop();
            html += `<p><a href="/videos/${magFile}" target="_blank">Download Magnified Video</a></p>`;
        }
        if (result.overlay_video) {
            const ovFile = result.overlay_video.split('/').pop();
            html += `<p><a href="/videos/${ovFile}" target="_blank">View Overlay Video</a></p>`;
        }
        resultContainer.innerHTML = html;
    }
}); 