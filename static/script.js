// Fixed recording functionality for the voice agent
document.addEventListener('DOMContentLoaded', function() {
    // Voice recording elements
    const recordBtn = document.getElementById('recordBtn');
    const resetBtn = document.getElementById('resetBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const resumeBtn = document.getElementById('resumeBtn');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const processingIndicator = document.getElementById('processingIndicator');
    const chatContainer = document.getElementById('chatContainer');
    const error = document.getElementById('error');
    
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    // Initialize with a client-side ID, but expect it to be overwritten by the server.
    let currentSessionId = 'session_init_' + Date.now();
    let websocket = null;
    let currentTranscriptDiv = null;
    let fallbackCheckInterval = null;
    // Track if we got a final transcript this WS session (prevents false fallback)
    let finalReceivedForSession = false;
    // For Day 21: accumulate streamed base64 audio chunks from server
    let audioBase64Chunks = [];
    // Minimal streaming acknowledgement flags
    // Log once per streaming session
    let streamAnnounced = false;
    // UI handles for streaming panel
    const audioStreamCard = document.getElementById('audioStreamCard');
    const streamStatus = document.getElementById('streamStatus');
    // Removed streamChunkCount and streamTotalChars
    const streamChunksList = document.getElementById('streamChunksList');
    const liveAudioEl = document.getElementById('liveAudio');

    // Control flags and audio graph refs
    let paused = false; // mic capture pause
    let audioContext = null; // capture context
    let source = null;
    let processor = null;
    let micStream = null;
    let turnEnded = false; // avoid handling turn_end twice

    // Playback pipeline via Web Audio API (append PCM parsed from WAV chunks)
    let playbackCtx = null;
    let playbackTimeCursor = 0; // seconds
    let wavHeaderParsed = false;
    let wavChannels = 1;
    let wavSampleRate = 44100;
    let wavBitsPerSample = 16;

    function setStreamVisible(visible) {
        if (!audioStreamCard) return;
        audioStreamCard.classList.toggle('hidden', !visible);
    }

    function setStreamStatus(state) {
        if (!streamStatus) return;
        streamStatus.textContent = state;
        streamStatus.classList.remove('pill-waiting', 'pill-active', 'pill-done');
        if (state === 'Streaming') streamStatus.classList.add('pill-active');
        else if (state === 'Completed') streamStatus.classList.add('pill-done');
        else streamStatus.classList.add('pill-waiting');
    }
    
    // Chat history functions
    function saveChatHistory(sessionId, messages) {
        localStorage.setItem(`chatHistory_${sessionId}`, JSON.stringify(messages));
    }
    
    function loadChatHistory(sessionId) {
        const stored = localStorage.getItem(`chatHistory_${sessionId}`);
        return stored ? JSON.parse(stored) : [];
    }
    
    function addMessageToHistory(sessionId, role, content) {
        const history = loadChatHistory(sessionId);
        history.push({
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        });
        saveChatHistory(sessionId, history);
        displayChatHistory(sessionId);
    }
    
    function displayChatHistory(sessionId) {
        const history = loadChatHistory(sessionId);
        chatContainer.innerHTML = '';
        
        history.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}`;
            messageDiv.textContent = msg.content;
            chatContainer.appendChild(messageDiv);
        });
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function showLiveTranscription(text, isFinal = false, sessionId = currentSessionId, turnOrder = null) {
        if (!currentTranscriptDiv) {
            currentTranscriptDiv = document.createElement('div');
            currentTranscriptDiv.className = 'message user live-transcript';
            currentTranscriptDiv.style.cssText = `
                background: linear-gradient(135deg, #e3f2fd, #bbdefb);
                border-left: 4px solid #2196F3;
                opacity: 0.8;
                font-style: italic;
            `;
            chatContainer.appendChild(currentTranscriptDiv);
        }
        
        currentTranscriptDiv.textContent = text;
        
        if (isFinal) {
            // If a final transcript for this session+turn already exists, update it instead of duplicating
            const turnSel = turnOrder !== null ? `[data-turn="${turnOrder}"]` : '';
            const existingFinal = document.querySelector(`.final-transcript[data-session-id="${sessionId}"]${turnSel}`);
            if (existingFinal) {
                existingFinal.textContent = text;
                // no history duplicate
                currentTranscriptDiv = null;
                chatContainer.scrollTop = chatContainer.scrollHeight;
                return;
            }
            currentTranscriptDiv.style.cssText = `
                background: linear-gradient(135deg, #f3e5f5, #e1bee7);
                border-left: 4px solid #9c27b0;
                opacity: 1;
                font-style: normal;
            `;
            currentTranscriptDiv.className = 'message user final-transcript';
            // Tag the element with the session ID and turn for dedupe
            currentTranscriptDiv.setAttribute('data-session-id', sessionId);
            if (turnOrder !== null) currentTranscriptDiv.setAttribute('data-turn', String(turnOrder));
            
            // Add to chat history
            addMessageToHistory(sessionId, 'user', text);
            
            // Reset for next transcription
            currentTranscriptDiv = null;
        }
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Safely tear down audio graph and close context without unhandled rejections
    async function teardownAudioGraph() {
        try { if (processor) { processor.disconnect(); } } catch {}
        try { if (source) { source.disconnect(); } } catch {}
        processor = null; source = null;
        try {
            if (micStream) {
                micStream.getTracks().forEach(t => { try { t.stop(); } catch {} });
            }
        } catch {}
        micStream = null;
        try {
            if (audioContext && audioContext.state !== 'closed') {
                await audioContext.close().catch(() => {});
            }
        } catch {}
        audioContext = null;
    }
    
    function showProcessingMessage(message) {
        // Remove any existing processing messages first
        const existingProcessingDiv = document.querySelector('.processing');
        if (existingProcessingDiv) {
            existingProcessingDiv.remove();
        }

        const processingDiv = document.createElement('div');
        processingDiv.className = 'message system processing';
        processingDiv.textContent = message;
        processingDiv.style.cssText = `
            background: linear-gradient(135deg, #fff3e0, #ffe0b2);
            color: #e65100;
            text-align: center;
            margin: 10px 0;
            padding: 12px;
            border-radius: 15px;
            border-left: 4px solid #ff9800;
            font-style: italic;
            animation: pulse 2s infinite;
        `;
        chatContainer.appendChild(processingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // Remove processing message after 15 seconds
        setTimeout(() => {
            if (processingDiv.parentNode) {
                processingDiv.remove();
            }
        }, 15000);
    }
    
    async function checkForRecentTranscriptions() {
        try {
            const response = await fetch('/recent-transcriptions');
            if (!response.ok) {
                console.error('Failed to fetch recent transcriptions:', response.statusText);
                return false;
            }
            const data = await response.json();
            
        if (data.transcriptions && data.transcriptions.length > 0) {
                // Check if any of the recent transcriptions match our session ID
                const foundTranscription = data.transcriptions.find(t => t.session_id === currentSessionId);

                // Check if a final transcript has already been rendered for this session
                const alreadyRendered = document.querySelector(`.final-transcript[data-session-id="${currentSessionId}"]`);

                if (foundTranscription && !alreadyRendered) {
                    console.log('📥 Fallback: Found transcription from server:', foundTranscription.text);
                    showLiveTranscription(foundTranscription.text, true, currentSessionId);
            // Ensure we don't show fallback error after WS close
            finalReceivedForSession = true;
                    
                    // Clear the interval since we found the transcription
                    if (fallbackCheckInterval) {
                        clearInterval(fallbackCheckInterval);
                        fallbackCheckInterval = null;
                    }
                    
                    // Remove processing message
                    const processingDiv = document.querySelector('.processing');
                    if (processingDiv) processingDiv.remove();
                    
                    return true;
                }
            }
        } catch (error) {
            console.error('Could not check for recent transcriptions:', error);
        }
        return false;
    }
    
    function resetConversation() {
        // Clear localStorage for current session
        localStorage.removeItem(`chatHistory_${currentSessionId}`);
        
        // Generate new session ID
        currentSessionId = 'session_init_' + Date.now();
        
        // Clear chat container
        chatContainer.innerHTML = '';
        
        // Clear any error messages
        const errorDiv = document.getElementById('error');
        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }
        
        // Show confirmation message
        const confirmDiv = document.createElement('div');
        confirmDiv.className = 'message system';
        confirmDiv.textContent = 'Conversation reset. Starting fresh!';
        confirmDiv.style.cssText = `
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            text-align: center;
            margin: 10px 0;
            padding: 12px;
            border-radius: 15px;
            animation: fadeIn 0.5s ease-in;
        `;
        chatContainer.appendChild(confirmDiv);
        
        // Remove confirmation message after 3 seconds
        setTimeout(() => {
            if (confirmDiv.parentNode) {
                confirmDiv.remove();
            }
        }, 3000);
        
        console.log('Conversation reset, new session ID:', currentSessionId);
    }
    
    function showError(message) {
        const errorDiv = document.getElementById('error');
        const errorMessage = document.getElementById('errorMessage');
        if (errorDiv && errorMessage) {
            errorMessage.textContent = message;
            errorDiv.classList.remove('hidden');
            setTimeout(() => errorDiv.classList.add('hidden'), 5000);
        }
        console.error('Error:', message);
    }
    
    // Initialize chat display
    displayChatHistory(currentSessionId);
    
    // Record button functionality
    if (recordBtn) {
        recordBtn.addEventListener('click', async () => {
            if (isRecording) {
                // Stop recording
                stopRecording();
            } else {
                // Start recording
                startRecording();
            }
        });
    }

    // Pause/Resume controls
    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            paused = true;
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send('pause');
            }
        });
    }
    if (resumeBtn) {
        resumeBtn.addEventListener('click', () => {
            paused = false;
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send('resume');
            }
        });
    }
    
    // Reset button functionality
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            // Add visual feedback
            resetBtn.style.transform = 'scale(0.95)';
            setTimeout(() => {
                resetBtn.style.transform = 'scale(1)';
            }, 150);
            
            // Reset the conversation
            resetConversation();
        });
    }
    
    async function startRecording() {
        try {
            // Establish WebSocket connection
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Use the dedicated turn-detection endpoint to keep concerns separated
            const wsUrl = `${protocol}//${window.location.host}/ws/turn-detection`;
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function() {
                console.log('WebSocket connection established');
            };
            
            websocket.onmessage = function(event) {
                console.log('Server response:', event.data);
                
                try {
                    const data = JSON.parse(event.data);
                    
                    switch(data.type) {
                        case 'connection_established':
                            console.log('✅ Streaming transcription ready');
                            // CAPTURE THE SESSION ID FROM THE SERVER
                            if (data.session_id) {
                                currentSessionId = data.session_id;
                                console.log('🔑 Session ID set by server:', currentSessionId);
                            }
                            // Reset flags for this new session
                            finalReceivedForSession = false;
                            turnEnded = false;
                            break;
                        // (audio_chunk and audio_stream_end handled below with full UI + playback)
                            
                        case 'transcribing':
                            console.log('⏳ Processing audio...');
                            showProcessingMessage(data.message);
                            break;
                            
                        case 'partial_transcript':
                            console.log('📝 Partial transcript:', data.text);
                            showLiveTranscription(data.text, false);
                            break;
                        case 'audio_chunk':
                            if (typeof data.data === 'string' && data.data.length) {
                                audioBase64Chunks.push(data.data);
                                if (!streamAnnounced) {
                                    console.log('Output streaming to client');
                                    streamAnnounced = true;
                                }
                                // Update UI
                                setStreamVisible(true);
                                setStreamStatus('Streaming');
                                if (streamChunksList) {
                                    const row = document.createElement('div');
                                    row.className = 'stream-chunk-row';
                                    row.textContent = `Chunk ${audioBase64Chunks.length}: ${data.data.slice(0, 64)}...`;
                                    streamChunksList.prepend(row);
                                }
                            }
                            break;
                        case 'audio_stream_end':
                            // Assemble the WAV into the audio element; do not auto-play
                            console.log('Output streaming complete. Preparing audio element...');
                            try {
                                if (liveAudioEl && audioBase64Chunks.length) {
                                    const b64 = audioBase64Chunks.join('');
                                    const src = `data:audio/wav;base64,${b64}`;
                                    liveAudioEl.src = src;
                                    try { liveAudioEl.load(); } catch {}
                                    const container = document.getElementById('liveAudioContainer');
                                    if (container) container.classList.remove('hidden');
                                    liveAudioEl.classList.remove('hidden');
                                    // Ensure the panel is visible even if no prior chunks toggled it
                                    setStreamVisible(true);
                                }
                            } catch (e) {
                                console.warn('Could not prepare audio element:', e);
                            }
                            setStreamStatus('Ready');
                            // Reset playback parsing state; keep chunks until after src is set
                            setTimeout(() => {
                                playbackTimeCursor = playbackCtx ? playbackCtx.currentTime : 0;
                                wavHeaderParsed = false;
                            }, 100);
                            // keep the list visible for context; reset chunk state for next turn after small delay
                            setTimeout(() => {
                                audioBase64Chunks = [];
                                streamAnnounced = false;
                            }, 1200);
                            break;
                        case 'paused':
                            console.log('⏸️ Server acknowledged pause');
                            break;
                        case 'resumed':
                            console.log('▶️ Server acknowledged resume');
                            break;
                            
                        case 'final_transcript':
                            console.log('✅ Final transcription received:', data.text);
                            // Stop any fallback checks that might have started
                            if (fallbackCheckInterval) {
                                clearInterval(fallbackCheckInterval);
                                fallbackCheckInterval = null;
                            }
                            // Mark that we do have a final transcript for this session
                            finalReceivedForSession = true;
                            showLiveTranscription(data.text, true, currentSessionId, (typeof data.turn_order === 'number' ? data.turn_order : null));
                            break;
                        
                        case 'turn_end':
                            if (!turnEnded) {
                                console.log('🛑 Turn ended by server.');
                                turnEnded = true;
                            }
                            // Do NOT close the WebSocket here; keep it open to receive audio chunks
                            // The server will send 'audio_stream_end' when TTS streaming completes
                            break;
                            
                        case 'transcription_error':
                            console.error('❌ Transcription error:', data.message);
                            showError('Transcription error: ' + data.message);
                            break;
                            
                        case 'audio_received':
                            // This can be noisy, so we can comment it out if not needed for debugging
                            // console.log(`📊 Audio chunk received: ${data.bytes} bytes`);
                            break;
                            
                        case 'error':
                            console.error('❌ WebSocket error:', data.message);
                            showError(data.message);
                            break;
                            
                        default:
                            console.log('Unknown message type:', data.type);
                    }
                } catch (e) {
                    // Handle non-JSON messages
                    console.log('Non-JSON message:', event.data);
                }
            };
            
            websocket.onclose = function(event) {
                console.log('WebSocket connection closed:', event.code, event.reason || '');
                // Stop polling if server closed cleanly
                if (fallbackCheckInterval) {
                    clearInterval(fallbackCheckInterval);
                    fallbackCheckInterval = null;
                }
                streamAnnounced = false;
                
                // If no final transcript exists, start fallback polling
                const alreadyRendered = document.querySelector(`.final-transcript[data-session-id="${currentSessionId}"]`);
                if (!alreadyRendered && !finalReceivedForSession) {
                    console.log('🔄 No final transcript found. Starting fallback check...');
                    let checkCount = 0;
                    const maxChecks = 10; // 10 checks * 2 seconds = 20 seconds timeout

                    fallbackCheckInterval = setInterval(async () => {
                        checkCount++;
                        const found = await checkForRecentTranscriptions();
                        if (found || checkCount >= maxChecks) {
                            clearInterval(fallbackCheckInterval);
                            fallbackCheckInterval = null;
                            if (!found) {
                                console.log('Fallback check timed out.');
                                const processingDiv = document.querySelector('.processing');
                                if (processingDiv) processingDiv.remove();
                                showError("Could not retrieve transcription. Please try again.");
                            }
                        }
                    }, 2000);
                }
                // Always reset audio chunks on close to avoid leaking state
                audioBase64Chunks = [];
            };

            websocket.onerror = function(error) {
                console.error('WebSocket error event:', error);
                showError('A connection error occurred. Please try again.');
            };

            // Get microphone access (we'll stream raw PCM 16k mono to server)
            micStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    channelCount: 1
                }
            });

            // Fallback MediaRecorder (used only if we need to upload a blob later)
            mediaRecorder = new MediaRecorder(micStream, { mimeType: 'audio/webm;codecs=opus' });
            audioChunks = [];
            mediaRecorder.ondataavailable = event => { if (event.data.size > 0) audioChunks.push(event.data); };

            // Live PCM streaming pipeline using Web Audio API
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            source = audioContext.createMediaStreamSource(micStream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                if (!(websocket && websocket.readyState === WebSocket.OPEN)) return;
                if (paused) return;
                const input = e.inputBuffer.getChannelData(0); // Float32 [-1,1]
                const pcm16 = new Int16Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    let s = Math.max(-1, Math.min(1, input[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
                }
                websocket.send(pcm16.buffer);
            };

            source.connect(processor);
            processor.connect(audioContext.destination);

            mediaRecorder.onstart = () => {
                console.log('Recording started');
                isRecording = true;
                recordBtn.classList.add('recording');
                const recordText = recordBtn.querySelector('.record-text');
                if (recordText) recordText.textContent = 'Stop Recording';
                recordingIndicator.classList.remove('hidden');
            };
            
            mediaRecorder.onstop = async () => {
                console.log('Recording stopped, waiting for transcription...');
                isRecording = false;
                recordBtn.classList.remove('recording');
                const recordText = recordBtn.querySelector('.record-text');
                if (recordText) recordText.textContent = 'Start Recording';
                recordingIndicator.classList.add('hidden');
                
                // Send a "stop_streaming" message to the server
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    console.log('Sending stop_streaming message');
                    websocket.send("stop_streaming");
                }
                
                // Start fallback timer: if no final transcript via WS within 3s, upload chunks
                const fallbackTimer = setTimeout(async () => {
                    // Cancel WS polling if we switch to upload fallback
                    if (fallbackCheckInterval) {
                        clearInterval(fallbackCheckInterval);
                        fallbackCheckInterval = null;
                    }
                    
                    const alreadyRendered = document.querySelector(`.final-transcript[data-session-id="${currentSessionId}"]`);
                    if (!alreadyRendered && audioChunks.length > 0) {
                        console.log('⛑️ Streaming fallback: uploading recorded audio to /agent/chat');
                        const completeBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                        try {
                            await processAudio(completeBlob);
                        } catch (e) {
                            console.error('Fallback upload failed:', e);
                        }
                    }
                }, 3000);
                
                // Teardown audio graph safely
                try { await teardownAudioGraph(); } catch {}
                
                // Finalize any pending transcription
                if (currentTranscriptDiv) {
                    currentTranscriptDiv.style.opacity = '1';
                    currentTranscriptDiv = null;
                }
            };
            
            // Start the fallback recorder to keep chunks for upload if needed
            mediaRecorder.start(1000);
            
        } catch (error) {
            console.error('Recording error:', error);
            let errorMessage = 'Could not start recording.';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Microphone access denied. Please allow access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No microphone found. Please connect a microphone.';
            }
            
            showError(errorMessage);
        }
    }
    
    async function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        
        // Send stop streaming message to server
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send('stop_streaming');
            // Wait for server to close, but force close after 1s as safety
            setTimeout(() => {
                if (websocket.readyState === WebSocket.OPEN) {
                    websocket.close(1000, 'Client stop');
                }
            }, 1000);
        }
        
        isRecording = false;
        recordBtn.classList.remove('recording');
        const recordText = recordBtn.querySelector('.record-text');
        if (recordText) recordText.textContent = 'Start Recording';
        recordingIndicator.classList.add('hidden');
        
        // Finalize any pending transcription
        if (currentTranscriptDiv) {
            currentTranscriptDiv.style.opacity = '1';
            currentTranscriptDiv = null;
        }
        // Tear down audio graph safely
        try { await teardownAudioGraph(); } catch {}
    }

    // ---------- Live playback helpers ----------
    function ensurePlaybackCtx() {
        if (!playbackCtx) {
            playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
            playbackTimeCursor = playbackCtx.currentTime;
        }
        return playbackCtx;
    }

    function b64ToUint8Array(b64) {
        const binary = atob(b64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    function findDataChunkOffset(bytes) {
        // Search for 'data' ASCII in the buffer
        for (let i = 0; i < bytes.length - 4; i++) {
            if (bytes[i] === 0x64 && bytes[i+1] === 0x61 && bytes[i+2] === 0x74 && bytes[i+3] === 0x61) {
                // data chunk size is next 4 bytes LE
                const size = bytes[i+4] | (bytes[i+5] << 8) | (bytes[i+6] << 16) | (bytes[i+7] << 24);
                return { offset: i + 8, size };
            }
        }
        return { offset: 44, size: bytes.length - 44 };
    }

    function parseWavHeaderIfNeeded(bytes) {
        if (wavHeaderParsed) return 0;
        // Basic RIFF check
        if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
            // channels
            wavChannels = bytes[22] | (bytes[23] << 8);
            wavSampleRate = bytes[24] | (bytes[25] << 8) | (bytes[26] << 16) | (bytes[27] << 24);
            wavBitsPerSample = bytes[34] | (bytes[35] << 8);
            const dataLoc = findDataChunkOffset(bytes);
            wavHeaderParsed = true;
            return dataLoc.offset;
        }
        // No header found, assume raw PCM
        wavHeaderParsed = true;
        return 0;
    }

    function schedulePcm(bytes) {
        const ctx = ensurePlaybackCtx();
        // Convert 16-bit PCM little-endian mono/stereo to Float32
        const bytesPerSample = wavBitsPerSample / 8;
        const frameCount = Math.floor(bytes.length / (bytesPerSample * wavChannels));
        const audioBuffer = ctx.createBuffer(wavChannels, frameCount, wavSampleRate);
        for (let ch = 0; ch < wavChannels; ch++) {
            const channel = audioBuffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i++) {
                const idx = (i * wavChannels + ch) * bytesPerSample;
                const lo = bytes[idx];
                const hi = bytes[idx + 1];
                // 16-bit signed
                let sample = (hi << 8) | lo;
                if (sample & 0x8000) sample = sample - 0x10000;
                channel[i] = sample / 0x8000;
            }
        }
        const src = ctx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(ctx.destination);
        const startAt = Math.max(ctx.currentTime + 0.02, playbackTimeCursor);
        try { src.start(startAt); } catch (e) { try { src.start(); } catch {} }
        playbackTimeCursor = startAt + audioBuffer.duration;
    }

    function handleIncomingAudioChunk(b64) {
        const bytes = b64ToUint8Array(b64);
        let offset = 0;
        if (!wavHeaderParsed) {
            offset = parseWavHeaderIfNeeded(bytes);
        }
        const pcmBytes = bytes.subarray(offset);
        if (pcmBytes.length) schedulePcm(pcmBytes);
        if (liveAudioEl && liveAudioEl.paused) {
            // Ensure some UI element is playing; link WebAudio output isn't via element, but keep for UX
            try { liveAudioEl.play().catch(() => {}); } catch {}
        }
    }
    
    async function processAudio(audioBlob) {
        if (processingIndicator) processingIndicator.classList.remove('hidden');
        
        try {
            const formData = new FormData();
            // The fallback blob is webm/opus; use a matching filename
            formData.append('file', audioBlob, 'recording.webm');
            
            const response = await fetch(`/agent/chat/${currentSessionId}`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok && data.status === 'success') {
                // Add user message to chat
                if (data.transcription) {
                    addMessageToHistory(currentSessionId, 'user', data.transcription);
                }
                
                // Add AI response to chat
                if (data.llm_response) {
                    addMessageToHistory(currentSessionId, 'assistant', data.llm_response);
                }
                
                // Prepare audio element (no autoplay)
        if (data.audio_url && liveAudioEl) {
                    try {
                        liveAudioEl.src = data.audio_url;
                        try { liveAudioEl.load(); } catch {}
                        const container = document.getElementById('liveAudioContainer');
                        if (container) container.classList.remove('hidden');
                        liveAudioEl.classList.remove('hidden');
            // Show the streaming panel with the ready audio element
            setStreamVisible(true);
            setStreamStatus('Ready');
                    } catch (e) {
                        console.log('Could not set audio URL on element:', e);
                    }
                }
                
            } else {
                showError(data.message || 'Failed to process audio');
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            showError('Failed to process audio. Please try again.');
        } finally {
            if (processingIndicator) processingIndicator.classList.add('hidden');
        }
    }
    
    // =============================
    // Separate Quick Transcribe UI
    // =============================
    (function initQuickTranscribe() {
        const startBtn = document.getElementById('transcribeStartBtn');
        const stopBtn = document.getElementById('transcribeStopBtn');
        const clearBtn = document.getElementById('transcribeClearBtn');
        const fileInput = document.getElementById('transcribeFileInput');
        const uploadBtn = document.getElementById('transcribeUploadBtn');
        const labels = document.getElementById('transcribeLabels');
        const status = document.getElementById('transcribeStatus');

        if (!startBtn || !stopBtn || !clearBtn || !fileInput || !uploadBtn || !labels) return;

        let rec;
        let recChunks = [];
        let recStream;

        function addLabel(text, variant = 'you') {
            const label = document.createElement('div');
            label.style.padding = '10px 12px';
            label.style.borderRadius = '12px';
            label.style.background = variant === 'you' ? '#e3f2fd' : '#ede7f6';
            label.style.borderLeft = variant === 'you' ? '4px solid #2196F3' : '4px solid #673AB7';
            label.style.color = '#222';
            label.style.fontSize = '14px';
            label.textContent = text;
            labels.appendChild(label);
        }

        function setStatus(msg) {
            if (status) status.textContent = msg || '';
        }

        async function transcribeBlob(blob) {
            setStatus('Uploading for transcription...');
            const fd = new FormData();
            fd.append('file', blob, 'quick-recording.webm');
            const res = await fetch('/transcribe/file', { method: 'POST', body: fd });
            const data = await res.json();
            if (res.ok && data.status === 'success') {
                // Display as sequential labels: split sentences roughly
                const parts = String(data.transcription || '').split(/(?<=[.!?])\s+/).filter(Boolean);
                if (!parts.length) addLabel('(no speech detected)', 'ai');
                parts.forEach((p, i) => {
                    setTimeout(() => addLabel(p.trim(), 'you'), i * 250);
                });
                setStatus('Transcription complete.');

                // Populate LLM output + audio element directly under Quick Transcribe
                const llmSec = document.getElementById('qtLLMSection');
                const llmText = document.getElementById('qtLLMText');
                const llmAudio = document.getElementById('qtLLMAudio');
                if (llmSec && llmText && llmAudio) {
                    llmSec.classList.remove('hidden');
                    llmText.textContent = data.llm_response || '(no LLM response)';
                    if (data.audio_url) {
                        llmAudio.src = data.audio_url;
                        // Attempt autoplay; if blocked, controls are visible
                        llmAudio.play().catch(() => {});
                    } else {
                        llmAudio.removeAttribute('src');
                    }
                }
            } else {
                addLabel('Transcription failed.', 'ai');
                setStatus(data.detail || data.message || 'Failed.');
            }
        }

        startBtn.addEventListener('click', async () => {
            try {
                recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                rec = new MediaRecorder(recStream, { mimeType: 'audio/webm;codecs=opus' });
                recChunks = [];
                rec.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
                rec.onstart = () => {
                    setStatus('Recording...');
                    startBtn.disabled = true; stopBtn.disabled = false;
                };
                rec.onstop = async () => {
                    setStatus('Finalizing recording...');
                    const blob = new Blob(recChunks, { type: 'audio/webm;codecs=opus' });
                    // Show a placeholder label to mimic progressive feel
                    addLabel('Processing your audio...', 'ai');
                    await transcribeBlob(blob);
                    startBtn.disabled = false; stopBtn.disabled = true;
                    if (recStream) recStream.getTracks().forEach(t => t.stop());
                };
                rec.start(1000);
            } catch (e) {
                setStatus('Mic error: ' + (e.message || e));
            }
        });

        stopBtn.addEventListener('click', () => {
            try { if (rec && rec.state !== 'inactive') rec.stop(); } catch {}
        });

        clearBtn.addEventListener('click', () => {
            labels.innerHTML = '';
            setStatus('');
        });

        uploadBtn.addEventListener('click', async () => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) { setStatus('Choose an audio file first.'); return; }
            try {
                addLabel('Processing selected file...', 'ai');
                await transcribeBlob(file);
            } catch (e) {
                setStatus('Upload error: ' + (e.message || e));
            }
        });
    })();
});
