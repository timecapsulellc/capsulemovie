document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generateBtn');
    const promptInput = document.getElementById('prompt');
    const previewSection = document.getElementById('previewSection');
    const progressBar = document.querySelector('#progressBar div');
    const statusText = document.getElementById('statusText');
    const previewVideo = document.getElementById('previewVideo');

    generateBtn.addEventListener('click', async function() {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }

        // Show preview section and reset progress
        previewSection.classList.remove('hidden');
        progressBar.style.width = '0%';
        statusText.textContent = 'Starting generation...';
        previewVideo.classList.add('hidden');

        try {
            // Start generation
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt }),
            });

            const data = await response.json();
            if (data.status === 'success') {
                // Poll for status updates
                pollStatus(data.job_id);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            statusText.textContent = `Error: ${error.message}`;
            progressBar.style.width = '0%';
        }
    });

    async function pollStatus(jobId) {
        try {
            const response = await fetch(`/api/status/${jobId}`);
            const data = await response.json();

            progressBar.style.width = `${data.progress}%`;
            statusText.textContent = `Progress: ${data.progress}%`;

            if (data.status === 'completed') {
                statusText.textContent = 'Generation completed!';
                previewVideo.src = data.video_url;
                previewVideo.classList.remove('hidden');
            } else if (data.status === 'processing') {
                // Continue polling
                setTimeout(() => pollStatus(jobId), 1000);
            } else {
                throw new Error('Generation failed');
            }
        } catch (error) {
            statusText.textContent = `Error: ${error.message}`;
        }
    }
});