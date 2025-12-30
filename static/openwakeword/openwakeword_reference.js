// main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const startButton = document.getElementById('start-button');
    const testButton = document.getElementById('test-button');
    const statusText = document.getElementById('status-text');
    const vadStatusIndicator = document.getElementById('vad-status');
    const detectionText = document.getElementById('detection-text');
    const chartCanvas = document.getElementById('score-chart');
    const micSelect = document.getElementById('mic-select');
    const debugAudioContainer = document.getElementById('debug-audio-container');
    const gainSlider = document.getElementById('gain-slider');
    const gainValue = document.getElementById('gain-value');

    // --- App State & Config ---
    let audioContext, workletNode, scoreChart, gainNode, mediaStream, successSound;
    let isListening = false;
    let detectionFadeTimer = null;
    let isDetectionCoolingDown = false; // Cooldown flag for success sound
    const sampleRate = 16000;
    const frameSize = 1280; // 80ms chunk size

    // --- VAD & Utterance Management ---
    // AHA Moment #6 (The Fix): The wake word model has a significant processing delay (latency)
    // compared to the VAD. The VAD hangover must be long enough to keep the "speech active"
    // window open long enough for the wake word score to peak. 8 frames was too short. 12 is robust.
    const VAD_HANGOVER_FRAMES = 12; // Increased from 8 to 12
    let vadHangoverCounter = 0;
    let isSpeechActive = false;
    let utteranceBuffer = [];

    // --- Pipeline Buffers & State ---
    let mel_buffer = [];
    let embedding_buffer = [];
    let vadState = { h: null, c: null };

    // --- Models ---
    let melspecModel, embeddingModel, vadModel;
    const models = {
        'alexa': { url: 'models/alexa_v0.1.onnx', session: null, scores: new Array(50).fill(0) },
        'hey_mycroft': { url: 'models/hey_mycroft_v0.1.onnx', session: null, scores: new Array(50).fill(0) },
        'hey_jarvis': { url: 'models/hey_jarvis_v0.1.onnx', session: null, scores: new Array(50).fill(0) },
    };

    const audioProcessorCode = `
        class AudioProcessor extends AudioWorkletProcessor {
            bufferSize = 1280;
            _buffer = new Float32Array(this.bufferSize);
            _pos = 0;
            constructor() { super(); }
            process(inputs) {
                const input = inputs[0][0];
                if (input) {
                    for (let i = 0; i < input.length; i++) {
                        this._buffer[this._pos++] = input[i];
                        if (this._pos === this.bufferSize) {
                            this.port.postMessage(this._buffer);
                            this._pos = 0;
                        }
                    }
                }
                return true;
            }
        }
        registerProcessor('audio-processor', AudioProcessor);
    `;


    // --- Charting & UI ---
    function initChart() {
        const chartContainer = chartCanvas.parentElement;
        if (chartContainer) {
            chartContainer.style.height = '240px';
            chartContainer.style.position = 'relative';
        }
        const ctx = chartCanvas.getContext('2d');
        scoreChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: 50 }, (_, i) => i + 1),
                datasets: Object.keys(models).map(name => ({
                    label: name,
                    data: models[name].scores,
                    borderColor: `hsl(${(Object.keys(models).indexOf(name) * 100)}, 70%, 50%)`,
                    borderWidth: 1,
                    fill: false,
                    tension: 0.1
                }))
            },
            options: { scales: { y: { beginAtZero: true, max: 1.0 } }, responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: 'top' } } }
        });
    }

    function updateChart() {
        scoreChart.data.datasets.forEach((d, i) => { d.data = models[Object.keys(models)[i]].scores; });
        scoreChart.update('none');
    }
    
    function injectCss() {
        const style = document.createElement('style');
        style.textContent = `
            #detection-text { opacity: 1; transition: opacity 1s ease-out; }
            #detection-text.fading { opacity: 0; }
        `;
        document.head.appendChild(style);
    }
    
    // --- WAV Audio Creation Helper ---
    function createWavBlobUrl(audioChunks) {
        let totalLength = audioChunks.reduce((len, chunk) => len + chunk.length, 0); if (totalLength === 0) return null;
        let combined = new Float32Array(totalLength); let offset = 0;
        for (const chunk of audioChunks) { combined.set(chunk, offset); offset += chunk.length; }
        let pcmData = new Int16Array(totalLength);
        for (let i = 0; i < totalLength; i++) { let s = Math.max(-1, Math.min(1, combined[i])); pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF; }
        const wavHeader = new ArrayBuffer(44); const view = new DataView(wavHeader); const channels = 1, bitsPerSample = 16;
        const byteRate = sampleRate * channels * (bitsPerSample / 8), blockAlign = channels * (bitsPerSample / 8);
        view.setUint32(0, 0x52494646, false); view.setUint32(4, 36 + pcmData.byteLength, true); view.setUint32(8, 0x57415645, false); view.setUint32(12, 0x666d7420, false);
        view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, channels, true); view.setUint32(24, sampleRate, true); view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true); view.setUint16(34, bitsPerSample, true); view.setUint32(36, 0x64617461, false); view.setUint32(40, pcmData.byteLength, true);
        const wavBlob = new Blob([view, pcmData], { type: 'audio/wav' }); return URL.createObjectURL(wavBlob);
    }

    // --- Model Loading & State Initialization ---
    async function loadModels() {
        statusText.textContent = 'Loading models...';
        const sessionOptions = { executionProviders: ['wasm'] };
        try {
            [melspecModel, embeddingModel, vadModel] = await Promise.all([ ort.InferenceSession.create('models/melspectrogram.onnx', sessionOptions), ort.InferenceSession.create('models/embedding_model.onnx', sessionOptions), ort.InferenceSession.create('models/silero_vad.onnx', sessionOptions) ]);
            for (const name in models) { models[name].session = await ort.InferenceSession.create(models[name].url, sessionOptions); }
            statusText.textContent = 'Models loaded. Ready to start.';
            startButton.disabled = false; testButton.disabled = false; startButton.textContent = 'Start Listening';
        } catch(e) {
            statusText.textContent = `Model loading failed: ${e.message}`;
            console.error(e);
        }
    }

    function resetState() {
        mel_buffer = [];
        embedding_buffer = [];
        for (let i = 0; i < 16; i++) {
            embedding_buffer.push(new Float32Array(96).fill(0));
        }
        const vadStateShape = [2, 1, 64];
        if (!vadState.h) {
            vadState.h = new ort.Tensor('float32', new Float32Array(128).fill(0), vadStateShape);
            vadState.c = new ort.Tensor('float32', new Float32Array(128).fill(0), vadStateShape);
        } else {
            vadState.h.data.fill(0);
            vadState.c.data.fill(0);
        }
        isSpeechActive = false;
        vadHangoverCounter = 0;
        utteranceBuffer = [];
        isDetectionCoolingDown = false;
        for (const name in models) { models[name].scores.fill(0); }
        detectionText.textContent = '';
        detectionText.classList.remove('fading');
        updateChart();
    }
    
    // --- VAD ---
    async function runVad(chunk) {
        try {
            const tensor = new ort.Tensor('float32', chunk, [1, chunk.length]);
            const sr = new ort.Tensor('int64', [BigInt(sampleRate)], []);
            const res = await vadModel.run({ input: tensor, sr: sr, h: vadState.h, c: vadState.c });
            vadState.h = res.hn; vadState.c = res.cn;
            return res.output.data[0] > 0.5;
        } catch (err) { console.error("VAD Error:", err); return false; }
    }

    // --- CONTINUOUS INFERENCE PIPELINE ---
    async function runInference(chunk, isSpeechConsideredActive) {
        // Stage 1: Audio Chunk -> Melspectrogram
        const melspecTensor = new ort.Tensor('float32', chunk, [1, frameSize]);
        const melspecResults = await melspecModel.run({ [melspecModel.inputNames[0]]: melspecTensor });
        let new_mel_data = melspecResults[melspecModel.outputNames[0]].data;

        // AHA Moment #1: The melspectrogram data MUST be transformed with this exact formula.
        for (let j = 0; j < new_mel_data.length; j++) { new_mel_data[j] = (new_mel_data[j] / 10.0) + 2.0; }
        
        for (let j = 0; j < 5; j++) {
            // AHA Moment #2: ONNX Runtime reuses output buffers. We MUST create a copy.
            mel_buffer.push(new Float32Array(new_mel_data.subarray(j * 32, (j + 1) * 32))); 
        }

        // Stage 2: Melspectrogram History -> Embedding Vector
        while (mel_buffer.length >= 76) {
            // AHA Moment #3: The embedding model requires a history of 76 spectrogram frames.
            const window_frames = mel_buffer.slice(0, 76);
            const flattened_mel = new Float32Array(76 * 32);
            for (let j = 0; j < window_frames.length; j++) { flattened_mel.set(window_frames[j], j * 32); }

            const embeddingFeeds = { [embeddingModel.inputNames[0]]: new ort.Tensor('float32', flattened_mel, [1, 76, 32, 1]) };
            const embeddingOut = await embeddingModel.run(embeddingFeeds);
            const new_embedding = embeddingOut[embeddingModel.outputNames[0]].data;

            // Stage 3: Embedding History -> Final Prediction
            // AHA Moment #4: The final wake word model also needs a history of the last 16 embeddings.
            embedding_buffer.shift();
            embedding_buffer.push(new Float32Array(new_embedding));

            const flattened_embeddings = new Float32Array(16 * 96);
            for (let j = 0; j < embedding_buffer.length; j++) { flattened_embeddings.set(embedding_buffer[j], j * 96); }
            const final_input_tensor = new ort.Tensor('float32', flattened_embeddings, [1, 16, 96]);

            for (const name in models) {
                const results = await models[name].session.run({ [models[name].session.inputNames[0]]: final_input_tensor });
                const score = results[models[name].session.outputNames[0]].data[0];
                models[name].scores.shift();
                models[name].scores.push(score);

                if (score > 0.5 && isSpeechConsideredActive && !isDetectionCoolingDown) {
                    isDetectionCoolingDown = true;
                    successSound.play();

                    if (detectionFadeTimer) clearTimeout(detectionFadeTimer);
                    detectionText.textContent = `Detected: ${name} (Score: ${score.toFixed(2)})`;
                    detectionText.classList.remove('fading');
                    detectionFadeTimer = setTimeout(() => { detectionText.classList.add('fading'); }, 1000);

                    setTimeout(() => { isDetectionCoolingDown = false; }, 2000); // 2-second cooldown
                }
            }
            
            mel_buffer.splice(0, 8);
        }
        updateChart();
    }

    // --- Audio Processing & UI Logic ---
    function stopListening() {
        if (isListening) {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }
            if (workletNode) { workletNode.port.onmessage = null; workletNode.disconnect(); workletNode = null; }
            if (gainNode) { gainNode.disconnect(); gainNode = null; }
            if (audioContext && audioContext.state !== 'closed') { audioContext.close(); }
            isListening = false;
        }
        startButton.textContent = 'Start Listening';
        statusText.textContent = 'Stopped.';
        vadStatusIndicator.classList.remove('active');
        isDetectionCoolingDown = false;
    }

    async function startListening() {
        if (isListening) return;
        
        startButton.disabled = true;
        startButton.textContent = 'Starting...';

        resetState();
        debugAudioContainer.innerHTML = '<h3>Debug Audio Clips</h3>';
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: micSelect.value ? { exact: micSelect.value } : undefined } });
            audioContext = new AudioContext({ sampleRate: sampleRate });
            const source = audioContext.createMediaStreamSource(mediaStream);
            
            gainNode = audioContext.createGain();
            gainNode.gain.value = parseFloat(gainSlider.value);
            
            const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
            const workletURL = URL.createObjectURL(blob);
            await audioContext.audioWorklet.addModule(workletURL);
            workletNode = new AudioWorkletNode(audioContext, 'audio-processor');

            workletNode.port.onmessage = async (event) => {
                const chunk = event.data; if (!chunk) return;
                
                const vadFired = await runVad(chunk);
                vadStatusIndicator.classList.toggle('active', vadFired);

                if (vadFired) {
                    if (!isSpeechActive) { utteranceBuffer = []; }
                    isSpeechActive = true;
                    vadHangoverCounter = VAD_HANGOVER_FRAMES;
                } else if (isSpeechActive) {
                    vadHangoverCounter--;
                    if (vadHangoverCounter <= 0) {
                        isSpeechActive = false;
                        const audioUrl = createWavBlobUrl(utteranceBuffer);
                        if (audioUrl) {
                            const clipContainer = document.createElement('div');
                            const clipTitle = document.createElement('p'); 
                            clipTitle.textContent = `Utterance at ${new Date().toLocaleTimeString()}:`;
                            const audioElement = document.createElement('audio'); 
                            audioElement.controls = true; 
                            audioElement.src = audioUrl;
                            clipContainer.appendChild(clipTitle);
                            clipContainer.appendChild(audioElement);
                            // Append to place new clips at the bottom, in order.
                            debugAudioContainer.appendChild(clipContainer);
                        }
                    }
                }

                if (isSpeechActive) {
                    utteranceBuffer.push(chunk);
                }
                
                await runInference(chunk, isSpeechActive);
            };

            source.connect(gainNode);
            gainNode.connect(workletNode);
            workletNode.connect(audioContext.destination);

            isListening = true; 
            startButton.textContent = 'Stop Listening';
            startButton.disabled = false;
            statusText.textContent = 'Listening...';
        } catch (err) { 
            console.error("Failed to start listening:", err); 
            statusText.textContent = `Error: ${err.message}`; 
            stopListening(); 
        }
    }

    // --- DIAGNOSTIC TEST ---
    async function runWavTest_DirectLogic() {
        if (isListening) stopListening();
        resetState();
        statusText.textContent = "[Direct Test] Running...";
        testButton.disabled = true;

        try {
            const resp = await fetch('./hey_jarvis_11-2.wav');
            const raw = await resp.arrayBuffer();
            const ac = new (window.AudioContext || window.webkitAudioContext)();
            const decoded = await ac.decodeAudioData(raw);
            const offline = new OfflineAudioContext(1, Math.ceil(decoded.length * 16000 / decoded.sampleRate), 16000);
            const src = offline.createBufferSource();
            src.buffer = decoded; src.connect(offline.destination); src.start();
            let audioData = (await offline.startRendering()).getChannelData(0);

            const minRequiredSamples = 16 * frameSize;
            if (audioData.length < minRequiredSamples) {
                const padding = new Float32Array(minRequiredSamples - audioData.length);
                const newAudioData = new Float32Array(minRequiredSamples);
                newAudioData.set(audioData, 0);
                newAudioData.set(padding, audioData.length);
                audioData = newAudioData;
            }

            let highestScore = 0.0;
            let local_mel_buffer = [];
            let local_embedding_buffer = [];
            for (let i = 0; i < 16; i++) { local_embedding_buffer.push(new Float32Array(96).fill(0)); }

            for (let i = 0; i < Math.floor(audioData.length / frameSize); i++) {
                const chunk = audioData.subarray(i * frameSize, (i + 1) * frameSize);

                const melspecFeeds = { [melspecModel.inputNames[0]]: new ort.Tensor('float32', chunk, [1, frameSize]) };
                const melspecOut = await melspecModel.run(melspecFeeds);
                const new_mel_data = melspecOut[melspecModel.outputNames[0]].data;

                for (let j = 0; j < new_mel_data.length; j++) { new_mel_data[j] = (new_mel_data[j] / 10.0) + 2.0; }
                for (let j = 0; j < 5; j++) { local_mel_buffer.push(new Float32Array(new_mel_data.subarray(j * 32, (j + 1) * 32))); }

                while (local_mel_buffer.length >= 76) {
                    const window_frames = local_mel_buffer.slice(0, 76);
                    const flattened_mel = new Float32Array(76 * 32);
                    for (let j = 0; j < window_frames.length; j++) { flattened_mel.set(window_frames[j], j * 32); }

                    const embeddingFeeds = { [embeddingModel.inputNames[0]]: new ort.Tensor('float32', flattened_mel, [1, 76, 32, 1]) };
                    const embeddingOut = await embeddingModel.run(embeddingFeeds);
                    const new_embedding = embeddingOut[embeddingModel.outputNames[0]].data;

                    local_embedding_buffer.shift();
                    local_embedding_buffer.push(new Float32Array(new_embedding));

                    const flattened_embeddings = new Float32Array(16 * 96);
                    for (let j = 0; j < local_embedding_buffer.length; j++) { flattened_embeddings.set(local_embedding_buffer[j], j * 96); }
                    const final_input_tensor = new ort.Tensor('float32', flattened_embeddings, [1, 16, 96]);
                    
                    const jarvisResults = await models['hey_jarvis'].session.run({ [models['hey_jarvis'].session.inputNames[0]]: final_input_tensor });
                    const currentScore = jarvisResults[models['hey_jarvis'].session.outputNames[0]].data[0];

                    if (currentScore > highestScore) { highestScore = currentScore; }
                    models['hey_jarvis'].scores.shift(); models['hey_jarvis'].scores.push(currentScore);
                    updateChart();

                    local_mel_buffer.splice(0, 8);
                }
            }
            statusText.textContent = `[Direct Test] Complete. Highest Score: ${highestScore.toFixed(6)}`;
            if (highestScore > 0.5) {
                detectionText.textContent = `[Direct Test] Detection SUCCESS!`;
                successSound.play();
            }

        } catch (e) { console.error("[Direct Test] failed", e); statusText.textContent = `Error: ${e.message}`;
        } finally { testButton.disabled = false; }
    }


    // --- Init ---
    async function init() {
        try {
            injectCss();
            initChart();
            successSound = new Audio('./success.mp3');
            successSound.preload = 'auto';
            await populateMicrophoneList();
            await loadModels();
        } catch (e) { console.error("Initialization failed", e); statusText.textContent = `Error: ${e.message}`; }
    }

    async function populateMicrophoneList() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioDevices = devices.filter(d => d.kind === 'audioinput');
            micSelect.innerHTML = '';
            audioDevices.forEach(d => { micSelect.add(new Option(d.label || `Mic ${micSelect.options.length + 1}`, d.deviceId)); });
        } catch (e) { console.error("Could not populate microphone list:", e); statusText.textContent = "Error: Could not access microphones."; }
    }

    gainSlider.addEventListener('input', () => {
        const gainLevel = parseFloat(gainSlider.value);
        if (gainNode) {
            gainNode.gain.value = gainLevel;
        }
        gainValue.textContent = `${Math.round(gainLevel * 100)}%`;
    });

    startButton.addEventListener('click', () => { if (isListening) { stopListening(); } else { startListening(); } });
    testButton.addEventListener('click', runWavTest_DirectLogic);
    
    init();
});