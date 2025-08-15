"""
BuddyBot - AI Voice Assistant
A conversational voice AI built with FastAPI, AssemblyAI, Google Gemini, and Murf AI
Simplified version with minimal dependencies and clean structure.
"""
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from io import BytesIO

# FastAPI imports
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# External service imports
import httpx
import assemblyai as aai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
class AppConfig:
    """Application configuration"""
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MURF_API_KEY = os.getenv("MURF_API_KEY")
    
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    MURF_API_URL = "https://api.murf.ai/v1/speech/generate"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    TRANSCRIPTION_TIMEOUT = 60
    LLM_TIMEOUT = 30
    TTS_TIMEOUT = 45
    MAX_LLM_RESPONSE_LENGTH = 3000

config = AppConfig()

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===== SERVICE INITIALIZATION =====
def initialize_services():
    """Initialize AI services"""
    # AssemblyAI
    if config.ASSEMBLYAI_API_KEY:
        aai.settings.api_key = config.ASSEMBLYAI_API_KEY
        logger.info("AssemblyAI configured successfully")
    
    # Google Gemini
    if config.GEMINI_API_KEY:
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("Google Gemini configured successfully")

initialize_services()

# ===== DATA MODELS =====
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationResponse(BaseModel):
    """Conversation response model"""
    session_id: str
    transcription: str
    llm_response: str
    audio_url: Optional[str] = None
    message_count: int
    status: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, str]
    message: str

# ===== CHAT HISTORY MANAGEMENT =====
class ChatManager:
    """Simple in-memory chat history manager"""
    
    def __init__(self):
        self._store: Dict[str, List[ChatMessage]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session"""
        if session_id not in self._store:
            self._store[session_id] = []
        
        message = ChatMessage(role=role, content=content, timestamp=datetime.now())
        self._store[session_id].append(message)
    
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get session history"""
        return self._store.get(session_id, [])
    
    def get_message_count(self, session_id: str) -> int:
        """Get message count for session"""
        return len(self._store.get(session_id, []))
    
    def clear_history(self, session_id: str) -> bool:
        """Clear session history"""
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

chat_manager = ChatManager()

# ===== AI SERVICES =====
async def transcribe_audio(audio_file) -> tuple[str, str]:
    """Transcribe audio using AssemblyAI"""
    try:
        if not config.ASSEMBLYAI_API_KEY:
            return "", "STT service not configured"
        
        config_obj = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
        transcriber = aai.Transcriber(config=config_obj)
        
        transcript = await asyncio.wait_for(
            asyncio.to_thread(transcriber.transcribe, audio_file),
            timeout=config.TRANSCRIPTION_TIMEOUT
        )
        
        if transcript.status == aai.TranscriptStatus.error:
            return "", f"Transcription error: {transcript.error}"
        
        if not transcript.text or not transcript.text.strip():
            return "", "No speech detected"
        
        return transcript.text, "success"
        
    except asyncio.TimeoutError:
        return "", "Transcription timeout"
    except Exception as e:
        return "", f"Transcription failed: {str(e)}"

async def generate_llm_response(text: str, chat_history: List[ChatMessage] = None) -> tuple[str, str]:
    """Generate LLM response using Google Gemini"""
    try:
        if not config.GEMINI_API_KEY:
            return "I'm having trouble connecting to my AI brain right now.", "LLM not configured"
        
        # Build context
        if chat_history:
            context = "You are a helpful AI assistant. Please respond conversationally and keep it under 2500 characters.\n\nConversation history:\n"
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            
            for message in recent_history:
                context += f"{message.role.title()}: {message.content}\n"
            
            context += f"User: {text}\n\nPlease respond:"
        else:
            context = f"You are a helpful AI assistant. Please respond to this conversationally (under 2500 characters): {text}"
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, context),
            timeout=config.LLM_TIMEOUT
        )
        
        if not response.text:
            return "I'm having trouble thinking right now.", "Empty LLM response"
        
        response_text = response.text.strip()
        
        # Truncate if too long
        if len(response_text) > config.MAX_LLM_RESPONSE_LENGTH:
            response_text = response_text[:2900] + "... I have more to share, but let me pause here."
        
        return response_text, "success"
        
    except asyncio.TimeoutError:
        return "I'm taking a bit longer to think. Let me give you a quick response for now.", "LLM timeout"
    except Exception as e:
        return "I'm having trouble processing your request right now.", f"LLM error: {str(e)}"

async def generate_speech(text: str, voice_id: str = "en-US-natalie") -> tuple[Optional[str], str]:
    """Generate speech using Murf AI"""
    try:
        if not config.MURF_API_KEY:
            return None, "TTS not configured"
        
        if not text or len(text) > 5000:
            return None, "Invalid text for TTS"
        
        headers = {
            "api-key": config.MURF_API_KEY.strip('"\''),
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "voiceId": voice_id,
            "format": "MP3",
            "sampleRate": 44100
        }
        
        async with httpx.AsyncClient(timeout=config.TTS_TIMEOUT) as client:
            response = await client.post(config.MURF_API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("audioFile"), "success"
            else:
                return None, f"TTS API error: {response.status_code}"
                
    except asyncio.TimeoutError:
        return None, "TTS timeout"
    except Exception as e:
        return None, f"TTS error: {str(e)}"

# ===== FALLBACK RESPONSES =====
FALLBACK_RESPONSES = {
    "stt_error": "I'm having trouble understanding your audio. Please try again.",
    "llm_error": "I'm having trouble thinking right now. Please try again in a moment.",
    "tts_error": "I understand you, but I'm having trouble generating speech right now.",
    "general_error": "I'm experiencing some technical difficulties. Please try again.",
    "no_speech": "I didn't hear anything. Could you speak louder or closer to your microphone?",
    "api_key_missing": "The service is temporarily unavailable. Please try again later."
}

async def generate_fallback_response(message: str, session_id: str = "error") -> dict:
    """Generate fallback response when services fail"""
    logger.warning(f"Generating fallback response: {message}")
    
    # Try to generate audio for error message
    audio_url, _ = await generate_speech(message, "en-US-ken")
    
    return {
        "session_id": session_id,
        "transcription": "System Error",
        "llm_response": message,
        "audio_url": audio_url,
        "message_count": 0,
        "status": "fallback"
    }

# ===== FASTAPI APPLICATION =====
app = FastAPI(
    title="BuddyBot - AI Voice Assistant",
    description="A conversational voice AI with speech-to-text, LLM, and text-to-speech capabilities.",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== ROUTES =====
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>BuddyBot</h1><p>Web interface not found. Please ensure index.html exists.</p>",
            status_code=500
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "speech_to_text": "available" if config.ASSEMBLYAI_API_KEY else "unavailable",
        "llm": "available" if config.GEMINI_API_KEY else "unavailable",
        "text_to_speech": "available" if config.MURF_API_KEY else "unavailable"
    }
    
    available_count = sum(1 for status in services.values() if status == "available")
    
    if available_count == 3:
        overall_status = "healthy"
        message = "All services operational"
    elif available_count > 0:
        overall_status = "degraded"
        message = "Some services may have limited functionality"
    else:
        overall_status = "unhealthy"
        message = "All services unavailable"
    
    return HealthResponse(status=overall_status, services=services, message=message)

@app.post("/agent/chat/{session_id}", response_model=ConversationResponse)
async def conversation_pipeline(session_id: str, file: UploadFile = File(...)):
    """Full conversational pipeline with session memory"""
    try:
        logger.info(f"Starting conversation for session {session_id}")
        
        # Step 1: Speech-to-Text
        transcribed_text, stt_status = await transcribe_audio(file.file)
        if stt_status != "success":
            return await generate_fallback_response(FALLBACK_RESPONSES["stt_error"], session_id)
        
        # Step 2: Add user message and get LLM response
        chat_manager.add_message(session_id, "user", transcribed_text)
        chat_history = chat_manager.get_history(session_id)
        
        llm_text, llm_status = await generate_llm_response(transcribed_text, chat_history)
        
        # Step 3: Add assistant message
        chat_manager.add_message(session_id, "assistant", llm_text)
        message_count = chat_manager.get_message_count(session_id)
        
        # Step 4: Text-to-Speech
        audio_url, tts_status = await generate_speech(llm_text)
        
        return ConversationResponse(
            session_id=session_id,
            transcription=transcribed_text,
            llm_response=llm_text,
            audio_url=audio_url,
            message_count=message_count,
            status="success" if tts_status == "success" else "partial_success"
        )
        
    except Exception as e:
        logger.error(f"Conversation error for session {session_id}: {str(e)}")
        return await generate_fallback_response(FALLBACK_RESPONSES["general_error"], session_id)

@app.get("/agent/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for session"""
    try:
        history = chat_manager.get_history(session_id)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in history
            ],
            "message_count": len(history)
        }
    except Exception as e:
        return {"session_id": session_id, "messages": [], "message_count": 0, "error": str(e)}

@app.delete("/agent/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for session"""
    success = chat_manager.clear_history(session_id)
    return {
        "session_id": session_id,
        "message": "Chat history cleared" if success else "No history found",
        "status": "success"
    }

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe audio file to text"""
    if not file.filename or not file.size:
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    if file.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    transcription, status = await transcribe_audio(file.file)
    
    if status == "success":
        return {"transcription": transcription, "status": "success"}
    else:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {status}")

@app.post("/generate-audio")
async def generate_audio_endpoint(request: dict):
    """Generate audio from text"""
    text = request.get("text", "")
    voice_id = request.get("voice_id", "en-US-natalie")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    audio_url, status = await generate_speech(text, voice_id)
    
    if status == "success":
        return {"audio_url": audio_url, "status": "success"}
    else:
        raise HTTPException(status_code=500, detail=f"TTS failed: {status}")

@app.post("/tts/echo")
async def tts_echo(file: UploadFile = File(...)):
    """Echo bot: transcribe and speak back"""
    transcription, stt_status = await transcribe_audio(file.file)
    
    if stt_status != "success":
        raise HTTPException(status_code=400, detail=f"Transcription failed: {stt_status}")
    
    audio_url, tts_status = await generate_speech(transcription)
    
    return {
        "transcription": transcription,
        "llm_response": transcription,
        "audio_url": audio_url,
        "status": "success" if tts_status == "success" else "partial_success"
    }

# ===== STARTUP =====
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("=" * 50)
    logger.info("BuddyBot - AI Voice Assistant Starting Up")
    logger.info("=" * 50)
    logger.info(f"Host: {config.HOST}:{config.PORT}")
    logger.info(f"AssemblyAI: {'Configured' if config.ASSEMBLYAI_API_KEY else 'Missing'}")
    logger.info(f"Gemini LLM: {'Configured' if config.GEMINI_API_KEY else 'Missing'}")
    logger.info(f"Murf TTS: {'Configured' if config.MURF_API_KEY else 'Missing'}")
    logger.info("BuddyBot is ready to chat!")
    logger.info(f"Open: http://{config.HOST}:{config.PORT}")
    logger.info("=" * 50)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting BuddyBot server...")
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
