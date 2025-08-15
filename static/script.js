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
    let currentSessionId = 'session_' + Date.now();
    
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
    
    function resetConversation() {
        // Clear localStorage for current session
        localStorage.removeItem(`chatHistory_${currentSessionId}`);
        
        // Generate new session ID
        currentSessionId = 'session_' + Date.now();
        
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
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                }
            });
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(track => track.stop());
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                if (audioBlob.size < 1000) {
                    showError('Recording too short. Please try again.');
                    return;
                }
                
                await processAudio(audioBlob);
            };
            
            mediaRecorder.start();
            isRecording = true;
            
            // Update UI
            recordBtn.classList.add('recording');
            const recordText = recordBtn.querySelector('.record-text');
            if (recordText) recordText.textContent = 'Stop Recording';
            recordingIndicator.classList.remove('hidden');
            
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
        
        isRecording = false;
        recordBtn.classList.remove('recording');
        const recordText = recordBtn.querySelector('.record-text');
        if (recordText) recordText.textContent = 'Start Recording';
        recordingIndicator.classList.add('hidden');
    }
    
    async function processAudio(audioBlob) {
        if (processingIndicator) processingIndicator.classList.remove('hidden');
        
        try {
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
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
});
