// Fixed recording functionality for the voice agent
document.addEventListener('DOMContentLoaded', function() {
    // Voice recording elements
    const recordBtn = document.getElementById('recordBtn');
    const resetBtn = document.getElementById('resetBtn');
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
    // For Day 21: accumulate streamed base64 audio chunks from server
    let audioBase64Chunks = [];
    // Minimal streaming acknowledgement flags
    // Log once per streaming session
    let streamAnnounced = false;
    // UI handles for streaming panel
    const audioStreamCard = document.getElementById('audioStreamCard');
    const streamStatus = document.getElementById('streamStatus');
    const streamChunkCount = document.getElementById('streamChunkCount');
    const streamTotalChars = document.getElementById('streamTotalChars');
    const streamChunksList = document.getElementById('streamChunksList');

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
    
    function showLiveTranscription(text, isFinal = false, sessionId = currentSessionId) {
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
            currentTranscriptDiv.style.cssText = `
                background: linear-gradient(135deg, #f3e5f5, #e1bee7);
                border-left: 4px solid #9c27b0;
                opacity: 1;
                font-style: normal;
            `;
            currentTranscriptDiv.className = 'message user final-transcript';
            // Tag the element with the session ID to prevent duplicates from the fallback
            currentTranscriptDiv.setAttribute('data-session-id', sessionId);
            
            // Add to chat history
            addMessageToHistory(sessionId, 'user', text);
            
            // Reset for next transcription
            currentTranscriptDiv = null;
        }
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
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
                            break;
                        case 'audio_chunk':
                            // Accumulate base64 chunks and log acknowledgement
                            if (typeof data.data === 'string' && data.data.length) {
                                audioBase64Chunks.push(data.data);
                                console.log(`🎧 Audio chunk received (${audioBase64Chunks.length})`);
                            }
                            break;
                        case 'audio_stream_end':
                            // Log final acknowledgement and size; do not auto-play
                            const totalChars1 = audioBase64Chunks.reduce((acc, s) => acc + s.length, 0);
                            console.log(`✅ Audio stream complete. Chunks: ${audioBase64Chunks.length}, total base64 chars: ${totalChars1}`);
                            // Reset for next turn
                            audioBase64Chunks = [];
                            break;
                            
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
                                if (streamChunkCount) streamChunkCount.textContent = String(audioBase64Chunks.length);
                                if (streamTotalChars) {
                                    const total = audioBase64Chunks.reduce((acc, s) => acc + s.length, 0);
                                    streamTotalChars.textContent = String(total);
                                }
                                if (streamChunksList) {
                                    const row = document.createElement('div');
                                    row.className = 'stream-chunk-row';
                                    row.textContent = `Chunk ${audioBase64Chunks.length}: ${data.data.slice(0, 64)}...`;
                                    streamChunksList.prepend(row);
                                }
                            }
                            break;
                        case 'audio_stream_end':
                            // Log final acknowledgement and size; do not auto-play
                            console.log('Output sent to client');
                            setStreamStatus('Completed');
                            // keep the list visible for context; reset counters for next turn after small delay
                            setTimeout(() => {
                                if (streamChunkCount) streamChunkCount.textContent = '0';
                                if (streamTotalChars) streamTotalChars.textContent = '0';
                                audioBase64Chunks = [];
                                streamAnnounced = false;
                            }, 1200);
                            break;
                            
                        case 'final_transcript':
                            console.log('✅ Final transcription received:', data.text);
                            // Stop any fallback checks that might have started
                            if (fallbackCheckInterval) {
                                clearInterval(fallbackCheckInterval);
                                fallbackCheckInterval = null;
                            }
                            showLiveTranscription(data.text, true);
                            break;
                        
                        case 'turn_end':
                            console.log('🛑 Turn ended by server.');
                            // Optionally show a subtle UI cue that turn has ended
                            // Close WebSocket after a short delay to finish any server cleanup
                            setTimeout(() => {
                                if (websocket && websocket.readyState === WebSocket.OPEN) {
                                    websocket.close(1000, 'Turn ended');
                                }
                            }, 300);
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
                if (!alreadyRendered) {
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
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    channelCount: 1
                }
            });

            // Fallback MediaRecorder (used only if we need to upload a blob later)
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
            audioChunks = [];
            mediaRecorder.ondataavailable = event => { if (event.data.size > 0) audioChunks.push(event.data); };

            // Live PCM streaming pipeline using Web Audio API
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                if (!(websocket && websocket.readyState === WebSocket.OPEN)) return;
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
                
                // Stop the microphone track
                const tracks = mediaRecorder.stream.getTracks();
                tracks.forEach(track => track.stop());
                
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
    
    function stopRecording() {
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
                
                // Play audio response if available
                if (data.audio_url) {
                    const audio = new Audio(data.audio_url);
                    audio.play().catch(e => console.log('Audio autoplay blocked'));
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
