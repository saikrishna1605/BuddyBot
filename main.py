
#add req imports 

from fastapi import FastAPI, UploadFile, File, Request, Path, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import requests
import os
import json
import logging
import asyncio
import httpx
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import google.generativeai as genai
from services.turn_detection import TurnDetectionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
MURF_KEY = os.getenv("MURF_API_KEY")
ASSEMBLY_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure APIs
if ASSEMBLY_KEY:
    aai.settings.api_key = ASSEMBLY_KEY
    logger.info("AssemblyAI configured successfully")
else:
    logger.warning("ASSEMBLYAI_API_KEY missing - speech recognition will fail")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Gemini configured successfully")
else:
    logger.warning("GEMINI_API_KEY missing - AI responses will fail")

if MURF_KEY:
    logger.info("Murf API key loaded successfully")
else:
    logger.warning("MURF_API_KEY missing - voice synthesis will fail")

# Configuration Constants
HOST = "localhost"
PORT = 8080
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
TRANSCRIPTION_TIMEOUT = 30
LLM_TIMEOUT = 30
TTS_TIMEOUT = 30
MAX_LLM_RESPONSE_LENGTH = 2000
MURF_API_URL = "https://api.murf.ai/v1/speech/generate"

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

# Global variable to store recent transcriptions
recent_transcriptions = []

def add_transcription_to_cache(text: str, session_id: str):
    """Add transcription to recent cache for fallback retrieval"""
    global recent_transcriptions
    transcription = {
        "text": text,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }
    recent_transcriptions.append(transcription)
    # Keep only last 10 transcriptions
    recent_transcriptions = recent_transcriptions[-10:]

chat_manager = ChatManager()

# ===== AI SERVICES =====
async def transcribe_audio(audio_file) -> tuple[str, str]:
    """Transcribe audio using AssemblyAI"""
    try:
        if not ASSEMBLY_KEY:
            return "", "STT service not configured"
        
        # Prepare an input path for the transcriber (it expects a path/URL)
        input_path: Optional[str] = None
        temp_path: Optional[Path] = None

        # If a string/path-like provided
        if isinstance(audio_file, (str, Path)):
            input_path = str(audio_file)
        else:
            # Assume it's a file-like; read bytes and write to a temp .webm file
            def _read_bytes(fobj):
                # Try to read from beginning
                try:
                    fobj.seek(0)
                except Exception:
                    pass
                return fobj.read()

            data: bytes = await asyncio.to_thread(_read_bytes, audio_file)
            if not data:
                return "", "Empty audio"

            # Write to temp file with a generic webm suffix (AssemblyAI supports webm/opus)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(data)
                temp_path = Path(tmp.name)
                input_path = tmp.name

        config_obj = aai.TranscriptionConfig(speech_model=aai.SpeechModel.best)
        transcriber = aai.Transcriber(config=config_obj)

        transcript = await asyncio.wait_for(
            asyncio.to_thread(transcriber.transcribe, input_path),
            timeout=TRANSCRIPTION_TIMEOUT
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
    finally:
        # Cleanup temp file if created
        try:
            if 'temp_path' in locals() and temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass

async def generate_llm_response(text: str, chat_history: List[ChatMessage] = None) -> tuple[str, str]:
    """Generate LLM response using Google Gemini"""
    try:
        if not GEMINI_API_KEY:
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
            timeout=LLM_TIMEOUT
        )
        
        if not response.text:
            return "I'm having trouble thinking right now.", "Empty LLM response"
        
        response_text = response.text.strip()
        
        # Truncate if too long
        if len(response_text) > MAX_LLM_RESPONSE_LENGTH:
            response_text = response_text[:2900] + "... I have more to share, but let me pause here."
        
        return response_text, "success"
        
    except asyncio.TimeoutError:
        return "I'm taking a bit longer to think. Let me give you a quick response for now.", "LLM timeout"
    except Exception as e:
        return "I'm having trouble processing your request right now.", f"LLM error: {str(e)}"

async def generate_speech(text: str, voice_id: str = "en-US-natalie") -> tuple[Optional[str], str]:
    """Generate speech using Murf AI"""
    try:
        if not MURF_KEY:
            return None, "TTS not configured"
        
        if not text or len(text) > 5000:
            return None, "Invalid text for TTS"
        
        headers = {
            "api-key": MURF_KEY.strip('"\''),
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "voiceId": voice_id,
            "format": "MP3",
            "sampleRate": 44100
        }
        
        async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
            response = await client.post(MURF_API_URL, json=payload, headers=headers)
            
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

app.mount("/static", StaticFiles(directory="static"), name="static")

import threading

# AssemblyAI Universal Streaming endpoint
@app.websocket("/ws/streaming")
async def streaming_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription with turn detection using Universal Streaming."""
    await websocket.accept()
    logger.info("Streaming WebSocket connection established")
    
    # Store the main event loop for thread-safe access
    main_loop = asyncio.get_running_loop()
    
    session_id = f"streaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not ASSEMBLY_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "AssemblyAI API key not configured"}))
        await websocket.close()
        return

    # Create a queue to communicate between event handlers and the main loop
    message_queue = asyncio.Queue()
    # Track latest partial transcript and whether a final was sent
    latest_transcript_text: Optional[str] = None
    got_final_transcript: bool = False

    # Create streaming client with the new Universal Streaming API
    streaming_client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLY_KEY,
            api_host="streaming.assemblyai.com"
        )
    )

    # Event handlers for the Universal Streaming API
    def on_begin(client, event: BeginEvent):
        logger.info(f"Streaming session started: {event.id}")

    def on_turn(client, event: TurnEvent):
        """Handle turn events with transcript data"""
        try:
            nonlocal latest_transcript_text, got_final_transcript
            logger.info(f"Turn event received - Turn order: {event.turn_order}, End of turn: {event.end_of_turn}, Transcript: {event.transcript}")
            
            # Send partial transcript while speaking
            if not event.end_of_turn and event.transcript:
                logger.info(f"Partial transcript: {event.transcript}")
                latest_transcript_text = event.transcript
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({
                        "type": "partial_transcript",
                        "text": event.transcript,
                        "session_id": session_id,
                        "turn_order": event.turn_order
                    }),
                    main_loop
                )
            
            # Send final transcript when turn ends
            elif event.end_of_turn and event.transcript:
                logger.info(f"Turn {event.turn_order} completed: {event.transcript}")
                got_final_transcript = True
                
                # Add to recent transcriptions for fallback mechanism
                recent_transcriptions.append({
                    "text": event.transcript,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                # Trim list if it gets too large
                if len(recent_transcriptions) > 100:
                    recent_transcriptions.pop(0)
                
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({
                        "type": "final_transcript",
                        "text": event.transcript,
                        "session_id": session_id,
                        "turn_order": event.turn_order,
                        "is_formatted": event.turn_is_formatted
                    }),
                    main_loop
                )

                # Explicitly notify client the turn ended
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({
                        "type": "turn_end",
                        "session_id": session_id,
                        "turn_order": event.turn_order
                    }),
                    main_loop
                )
                
            # Handle case where we have transcript but no explicit end of turn
            elif event.transcript and not hasattr(event, 'end_of_turn'):
                logger.info(f"Transcript received without clear turn end: {event.transcript}")
                latest_transcript_text = event.transcript
                # Add to recent transcriptions for fallback
                recent_transcriptions.append({
                    "text": event.transcript,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({
                        "type": "partial_transcript",
                        "text": event.transcript,
                        "session_id": session_id,
                        "turn_order": event.turn_order
                    }),
                    main_loop
                )
                
        except Exception as e:
            logger.error(f"Error handling turn event: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def on_terminated(client, event: TerminationEvent):
        logger.info(f"Streaming session terminated: {event.audio_duration_seconds} seconds processed")

    def on_error(client, error: StreamingError):
        logger.error(f"Universal Streaming error: {error}")
        try:
            asyncio.run_coroutine_threadsafe(
                message_queue.put({
                    "type": "error",
                    "message": str(error)
                }),
                main_loop
            )
        except RuntimeError as e:
            logger.error(f"Could not send error to queue (no event loop): {e}")
        except Exception as e:
            logger.error(f"Unexpected error in error handler: {e}")

    # Register event handlers
    streaming_client.on(StreamingEvents.Begin, on_begin)
    streaming_client.on(StreamingEvents.Turn, on_turn)
    streaming_client.on(StreamingEvents.Termination, on_terminated)
    streaming_client.on(StreamingEvents.Error, on_error)

    try:
        # Connect to AssemblyAI Universal Streaming
        # We stream raw PCM16 at 16kHz to the server (matches client-side WebAudio pipeline)
        streaming_client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                encoding="pcm_s16le"
            )
        )
        
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Universal Streaming transcription with turn detection ready",
            "session_id": session_id
        }))

        # Create tasks for handling audio input and message output
        async def handle_client_audio():
            """Handle audio streaming from client"""
            finalization_timeout = 3.0  # reserved: seconds of silence before finalizing (not used yet)
            
            while True:
                try:
                    message = await websocket.receive()
                    if "bytes" in message:
                        # Stream audio data to AssemblyAI
                        streaming_client.stream(message["bytes"])
                        
                        # Send audio received confirmation (optional, for debugging)
                        await websocket.send_text(json.dumps({
                            "type": "audio_received",
                            "bytes": len(message["bytes"]),
                            "session_id": session_id
                        }))
                        
                    elif "text" in message and message["text"] == "stop_streaming":
                        logger.info("Client requested to stop streaming.")
                        # If no final transcript was produced, emit the latest partial as final
                        if not got_final_transcript and latest_transcript_text:
                            logger.info(f"Finalizing transcript on stop (fallback): {latest_transcript_text}")
                            recent_transcriptions.append({
                                "text": latest_transcript_text,
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat()
                            })
                            await message_queue.put({
                                "type": "final_transcript",
                                "text": latest_transcript_text,
                                "session_id": session_id,
                                "turn_order": 1
                            })
                        # Terminate streaming session gracefully
                        try:
                            streaming_client.disconnect(terminate=True)
                        except Exception:
                            pass
                        # Gracefully close the websocket
                        try:
                            await websocket.close(code=1000)
                        except Exception:
                            pass
                        break
                except WebSocketDisconnect:
                    logger.info("Client disconnected during streaming.")
                    break
                except Exception as e:
                    logger.error(f"Error receiving audio: {e}")
                    break

        async def handle_transcript_messages():
            """Handle messages from AssemblyAI and send to client"""
            message_count = 0
            while True:
                try:
                    # Wait for messages from the event handlers
                    message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
                    message_count += 1
                    logger.info(f"Sending message #{message_count} to client: {message['type']}")
                    await websocket.send_text(json.dumps(message))
                    
                    # If this is a final transcript, we can break the loop
                    if message.get('type') == 'final_transcript':
                        logger.info("Final transcript sent, ending message handler")
                        break
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error sending transcript message: {e}")
                    break

        # Run both tasks concurrently
        await asyncio.gather(
            handle_client_audio(),
            handle_transcript_messages()
        )

    except Exception as e:
        logger.error(f"Universal Streaming WebSocket error: {e}")
    finally:
        # Cleanly close the streaming connection
        try:
            streaming_client.disconnect(terminate=True)
        except:
            pass
        logger.info("Universal Streaming WebSocket connection closed")


# Dedicated, minimal turn-detection-only WebSocket endpoint
@app.websocket("/ws/turn-detection")
async def turn_detection_ws(websocket: WebSocket):
    """Separate endpoint for Day 18: Turn Detection (no overlap with main pipeline)."""
    if not ASSEMBLY_KEY:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "AssemblyAI API key not configured"
        }))
        await websocket.close()
        return

    service = TurnDetectionService(api_key=ASSEMBLY_KEY)
    # Delegate all handling to the focused service
    await service.stream_handler(websocket, ASSEMBLY_KEY)
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
        "speech_to_text": "available" if ASSEMBLY_KEY else "unavailable",
        "llm": "available" if GEMINI_API_KEY else "unavailable",
        "text_to_speech": "available" if MURF_KEY else "unavailable"
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
        
        
        transcribed_text, stt_status = await transcribe_audio(file.file)
        if stt_status != "success":
            return await generate_fallback_response(FALLBACK_RESPONSES["stt_error"], session_id)
        
        chat_manager.add_message(session_id, "user", transcribed_text)
        chat_history = chat_manager.get_history(session_id)
        
        llm_text, llm_status = await generate_llm_response(transcribed_text, chat_history)
        
        chat_manager.add_message(session_id, "assistant", llm_text)
        message_count = chat_manager.get_message_count(session_id)
        
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

@app.get("/recent-transcriptions")
async def get_recent_transcriptions():
    """Get recent transcriptions as fallback when WebSocket fails"""
    global recent_transcriptions
    return {
        "transcriptions": recent_transcriptions,
        "count": len(recent_transcriptions),
        "message": "Recent transcription results"
    }

@app.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe audio file to text"""
    if not file.filename or not file.size:
        raise HTTPException(status_code=400, detail="Invalid audio file")
    
    if file.size > MAX_FILE_SIZE:
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
    logger.info(f"Host: {HOST}:{PORT}")
    logger.info(f"AssemblyAI: {'Configured' if ASSEMBLY_KEY else 'Missing'}")
    logger.info(f"Gemini LLM: {'Configured' if GEMINI_API_KEY else 'Missing'}")
    logger.info(f"Murf TTS: {'Configured' if MURF_KEY else 'Missing'}")
    logger.info("BuddyBot is ready to chat!")
    logger.info(f"Open: http://{HOST}:{PORT}")
    logger.info("=" * 50)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting BuddyBot server...")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False
    )
