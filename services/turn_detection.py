import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional, Callable
import threading
import contextlib

import websockets

import google.generativeai as genai

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

logger = logging.getLogger(__name__)
MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"
MURF_KEY = os.getenv("MURF_API_KEY")

class TurnDetectionService:
    """Encapsulates AssemblyAI Universal Streaming turn detection."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_client(self) -> StreamingClient:
        return StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

    def connect(self, client: StreamingClient) -> None:
        params = StreamingParameters(
            sample_rate=16000,
            format_turns=True,
            encoding="pcm_s16le",
        )
        logger.info(
            f"[TurnDetection] Connecting with encoding={params.encoding}, sample_rate={params.sample_rate}"
        )
        client.connect(params)

    def wire_handlers(
        self,
        client: StreamingClient,
        loop: asyncio.AbstractEventLoop,
        session_id: str,
        send_queue: asyncio.Queue,
        on_final: Optional[Callable[[str], None]] = None,
    ) -> None:
        latest_partial: Optional[str] = None
        got_final = False

        def on_begin(_c, event: BeginEvent):
            logger.info(f"[TurnDetection] Begin: {event.id}")

        def on_turn(_c, event: TurnEvent):
            nonlocal latest_partial, got_final
            try:
                if not event.end_of_turn and event.transcript:
                    latest_partial = event.transcript
                    asyncio.run_coroutine_threadsafe(
                        send_queue.put({
                            "type": "partial_transcript",
                            "text": event.transcript,
                            "session_id": session_id,
                            "turn_order": event.turn_order,
                        }),
                        loop,
                    )
                elif event.end_of_turn and event.transcript:
                    got_final = True
                    final_text = event.transcript
                    asyncio.run_coroutine_threadsafe(
                        send_queue.put({
                            "type": "final_transcript",
                            "text": final_text,
                            "session_id": session_id,
                            "turn_order": event.turn_order,
                            "is_formatted": event.turn_is_formatted,
                        }),
                        loop,
                    )
                    asyncio.run_coroutine_threadsafe(
                        send_queue.put({
                            "type": "turn_end",
                            "session_id": session_id,
                            "turn_order": event.turn_order,
                        }),
                        loop,
                    )
                    if on_final:
                        on_final(final_text)
                elif event.transcript and not hasattr(event, "end_of_turn"):
                    latest_partial = event.transcript
                    asyncio.run_coroutine_threadsafe(
                        send_queue.put({
                            "type": "partial_transcript",
                            "text": event.transcript,
                            "session_id": session_id,
                            "turn_order": event.turn_order,
                        }),
                        loop,
                    )
            except Exception as e:
                logger.exception(f"[TurnDetection] on_turn error: {e}")

        def on_terminated(_c, event: TerminationEvent):
            logger.info(
                f"[TurnDetection] Terminated, duration: {event.audio_duration_seconds} s"
            )

        def on_error(_c, error: StreamingError):
            logger.error(f"[TurnDetection] Error: {error}")
            asyncio.run_coroutine_threadsafe(
                send_queue.put({"type": "error", "message": str(error)}),
                loop,
            )

        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)

    async def stream_handler(self, websocket, api_key: str):
        """A minimal, focused handler for Day 18 turn detection demo."""
        await websocket.accept()
        session_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        loop = asyncio.get_running_loop()
        send_queue: asyncio.Queue = asyncio.Queue()
        # Murf streaming queue to forward LLM chunks to TTS
        murf_queue: asyncio.Queue = asyncio.Queue()
        murf_task: Optional[asyncio.Task] = None

        client = self.create_client()

        async def murf_ws_worker(static_ctx: str):
            """Open a Murf WS, forward text chunks from murf_queue, and print base64 audio."""
            if not MURF_KEY:
                logger.warning("[TurnDetection] MURF_API_KEY missing; skipping Murf WS TTS")
                return
            qs = f"?api-key={MURF_KEY.strip('\"\'')}\u0026sample_rate=44100\u0026channel_type=MONO\u0026format=WAV"
            url = f"{MURF_WS_URL}{qs}"
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                    voice_cfg = {
                        "voice_config": {
                            "voiceId": "en-US-amara",
                            "style": "Conversational",
                            "rate": 0,
                            "pitch": 0,
                            "variation": 1,
                        }
                    }
                    await ws.send(json.dumps(voice_cfg))

                    async def receiver():
                        print("\n--- Murf WS audio stream (base64) ---")
                        try:
                            while True:
                                raw = await ws.recv()
                                try:
                                    data = json.loads(raw)
                                except Exception:
                                    logger.debug(f"[TurnDetection] Murf WS non-JSON: {raw}")
                                    continue
                                if isinstance(data, dict):
                                    if "audio" in data:
                                        print(data["audio"])  # base64
                                    if data.get("final"):
                                        break
                        finally:
                            print("--- Murf WS audio stream end ---\n")

                    recv_task = asyncio.create_task(receiver())
                    # Sender loop: forward chunks
                    while True:
                        item = await murf_queue.get()
                        if isinstance(item, dict) and item.get("end"):
                            await ws.send(json.dumps({"context_id": static_ctx, "end": True}))
                            break
                        if isinstance(item, dict) and "text" in item:
                            await ws.send(json.dumps({"context_id": static_ctx, "text": item["text"]}))
                    # Wait for receiver to finish after final
                    with contextlib.suppress(Exception):
                        await recv_task
            except Exception as e:
                logger.warning(f"[TurnDetection] Murf WS worker error: {e}")

        def stream_llm_response(text: str) -> None:
            try:
                if not text or not text.strip():
                    return
                prompt = (
                    "You are a helpful assistant. Answer the user's request clearly, concisely, and conversationally.\n"
                    f"User said: \"{text}\"\nRespond directly to the user."
                )
                logger.info("[TurnDetection] LLM streaming input text: %s", text[:300])
                model = genai.GenerativeModel('gemini-1.5-flash')
                stream = model.generate_content(prompt, stream=True)
                chunks = []
                print("\n=== Streaming LLM response (Gemini) ===")
                for chunk in stream:
                    part = getattr(chunk, 'text', '') or ''
                    if part:
                        print(part, end='', flush=True)
                        chunks.append(part)
                        # Forward this chunk to Murf WS via the async queue
                        try:
                            asyncio.run_coroutine_threadsafe(murf_queue.put({"text": part}), loop)
                        except Exception as fe:
                            logger.debug(f"[TurnDetection] could not enqueue Murf text: {fe}")
                print("\n=== End of LLM stream ===\n")
                try:
                    stream.resolve()
                except Exception:
                    pass
                full_text = ''.join(chunks).strip()
                logger.info("[TurnDetection] LLM streamed response length: %d", len(full_text))
                # Signal end to Murf so it can finalize
                try:
                    asyncio.run_coroutine_threadsafe(murf_queue.put({"end": True}), loop)
                except Exception:
                    pass
            except Exception as e:
                logger.exception(f"[TurnDetection] LLM streaming error: {e}")

        def on_final(text: str):
            # Start Murf worker with a static context_id to avoid context limit issues
            nonlocal murf_task
            static_ctx = f"{session_id}_ctx"
            if not murf_task or murf_task.done():
                murf_task = asyncio.create_task(murf_ws_worker(static_ctx))
            # Launch LLM streaming in a background thread; chunks will be forwarded to Murf
            threading.Thread(target=stream_llm_response, args=(text,), daemon=True).start()

        self.wire_handlers(client, loop, session_id, send_queue, on_final)
        self.connect(client)

        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection_established",
                    "message": "Turn detection ready",
                    "session_id": session_id,
                }
            )
        )

        async def recv_audio():
            while True:
                msg = await websocket.receive()
                if "bytes" in msg:
                    client.stream(msg["bytes"])
                elif "text" in msg and msg["text"] == "stop_streaming":
                    try:
                        client.disconnect(terminate=True)
                    except Exception:
                        pass
                    try:
                        await websocket.close(code=1000)
                    except Exception:
                        pass
                    break

        async def send_messages():
            while True:
                try:
                    payload = await asyncio.wait_for(send_queue.get(), timeout=2.0)
                    await websocket.send_text(json.dumps(payload))
                    if payload.get("type") == "turn_end":
                        # end after the turn for the demo
                        break
                except asyncio.TimeoutError:
                    continue

        try:
            await asyncio.gather(recv_audio(), send_messages())
        finally:
            try:
                client.disconnect(terminate=True)
            except Exception:
                pass
            # Close Murf worker if still running
            if murf_task and not murf_task.done():
                murf_task.cancel()
                with contextlib.suppress(Exception):
                    await murf_task
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info("[TurnDetection] WebSocket closed")
