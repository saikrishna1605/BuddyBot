import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable
import threading

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

        client = self.create_client()

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
                print("\n=== End of LLM stream ===\n")
                try:
                    stream.resolve()
                except Exception:
                    pass
                full_text = ''.join(chunks).strip()
                logger.info("[TurnDetection] LLM streamed response length: %d", len(full_text))
            except Exception as e:
                logger.exception(f"[TurnDetection] LLM streaming error: {e}")

        def on_final(text: str):
            # Launch streaming in a background thread to avoid blocking event loop / handlers
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
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info("[TurnDetection] WebSocket closed")
