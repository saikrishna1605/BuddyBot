import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional, Callable
import threading
import contextlib
import concurrent.futures

import websockets

import google.generativeai as genai
from starlette.websockets import WebSocketDisconnect
from services.transcription_cache import add_transcription_to_cache
from services.skills import handle_weather_query, handle_stock_query
from services.search import web_search, format_search_summary, speechify_summary
from services.config import config as app_config

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

class TurnDetectionService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.murf_key = (app_config.get("MURF_API_KEY") or "").strip('\"\'')

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
        state: Optional[dict] = None,
    ) -> None:
        latest_partial: Optional[str] = None
        if state is not None and "last_final_turn" not in state:
            state["last_final_turn"] = None

        def on_begin(_c, event: BeginEvent):
            logger.info(f"[TurnDetection] Begin: {event.id}")

        def on_turn(_c, event: TurnEvent):
            nonlocal latest_partial
            try:
                if not event.end_of_turn and event.transcript:
                    latest_partial = event.transcript
                    if state is not None:
                        state["latest_partial"] = latest_partial
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
                    if state is not None and state.get("last_final_turn") == event.turn_order:
                        return
                    if state is not None:
                        state["got_final"] = True
                    final_text = event.transcript
                    if state is not None:
                        state["last_final_turn"] = event.turn_order
                    try:
                        add_transcription_to_cache(final_text, session_id)
                    except Exception:
                        pass
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
        await websocket.accept()
        session_id = f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        loop = asyncio.get_running_loop()
        send_queue: asyncio.Queue = asyncio.Queue()
        murf_queue: asyncio.Queue = asyncio.Queue()
        murf_future: Optional[concurrent.futures.Future] = None
        paused: bool = False
        use_search: bool = False
        memory: list = []

        client = self.create_client()

        async def murf_ws_worker(static_ctx: str):
            self.murf_key = (app_config.get("MURF_API_KEY") or "").strip('\"\'')
            if not self.murf_key:
                logger.warning("[TurnDetection] MURF_API_KEY missing; skipping Murf WS TTS")
                return
            qs = f"?api-key={self.murf_key}\u0026sample_rate=44100\u0026channel_type=MONO\u0026format=WAV"
            url = f"{MURF_WS_URL}{qs}"
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                    voice_cfg = {"voice_config": {"voiceId": "en-US-amara", "style": "Conversational", "rate": 0, "pitch": 0, "variation": 1}}
                    await ws.send(json.dumps(voice_cfg))
                    async def receiver():
                        try:
                            while True:
                                raw = await ws.recv()
                                try:
                                    data = json.loads(raw)
                                except Exception:
                                    continue
                                if isinstance(data, dict):
                                    if "audio" in data:
                                        await send_queue.put({"type": "audio_chunk", "data": data["audio"], "session_id": session_id})
                                    if data.get("final"):
                                        await send_queue.put({"type": "audio_stream_end", "session_id": session_id})
                                        break
                        finally:
                            return
                    recv_task = asyncio.create_task(receiver())
                    while True:
                        item = await murf_queue.get()
                        if isinstance(item, dict) and item.get("end"):
                            await ws.send(json.dumps({"context_id": static_ctx, "end": True}))
                            break
                        if isinstance(item, dict) and "text" in item:
                            await ws.send(json.dumps({"context_id": static_ctx, "text": item["text"]}))
                    with contextlib.suppress(Exception):
                        await recv_task
            except Exception as e:
                logger.warning(f"[TurnDetection] Murf WS worker error: {e}")

        def stream_llm_response(text: str) -> None:
            try:
                if not text or not text.strip():
                    return
                try:
                    memory.append({"role": "user", "content": text})
                    if len(memory) > 16:
                        del memory[0:len(memory)-16]
                except Exception:
                    pass
                try:
                    wx = asyncio.run(handle_weather_query(text))
                except RuntimeError:
                    loop2 = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop2)
                    wx = loop2.run_until_complete(handle_weather_query(text))
                    loop2.close()
                except Exception:
                    wx = None
                if wx:
                    asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_stream_start", "session_id": session_id, "tts_unavailable": False if self.murf_key else True}), loop)
                    asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_delta", "text": wx, "session_id": session_id}), loop)
                    if self.murf_key:
                        asyncio.run_coroutine_threadsafe(murf_queue.put({"text": wx}), loop)
                        asyncio.run_coroutine_threadsafe(murf_queue.put({"end": True}), loop)
                    asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_final", "text": wx, "session_id": session_id}), loop)
                    return
                try:
                    stock = asyncio.run(handle_stock_query(text, chat_history=memory))
                except RuntimeError:
                    loop2 = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop2)
                    stock = loop2.run_until_complete(handle_stock_query(text, chat_history=memory))
                    loop2.close()
                except Exception:
                    stock = None
                if stock:
                    asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_stream_start", "session_id": session_id, "tts_unavailable": False if self.murf_key else True}), loop)
                    asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_delta", "text": stock, "session_id": session_id}), loop)
                    if self.murf_key:
                        asyncio.run_coroutine_threadsafe(murf_queue.put({"text": stock}), loop)
                        asyncio.run_coroutine_threadsafe(murf_queue.put({"end": True}), loop)
                    asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_final", "text": stock, "session_id": session_id}), loop)
                    try:
                        memory.append({"role": "assistant", "content": stock})
                        if len(memory) > 16:
                            del memory[0:len(memory)-16]
                    except Exception:
                        pass
                    return
                if use_search:
                    try:
                        try:
                            data, err = asyncio.run(web_search(text))
                        except RuntimeError:
                            loop2 = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop2)
                            data, err = loop2.run_until_complete(web_search(text))
                            loop2.close()
                        if not err and data:
                            answer, sources = format_search_summary(data)
                            asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_stream_start", "session_id": session_id, "tts_unavailable": False if self.murf_key else True, "mode": "search"}), loop)
                            speech = speechify_summary(answer)
                            asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_delta", "text": speech, "session_id": session_id}), loop)
                            if self.murf_key:
                                asyncio.run_coroutine_threadsafe(murf_queue.put({"text": speech}), loop)
                                asyncio.run_coroutine_threadsafe(murf_queue.put({"end": True}), loop)
                            asyncio.run_coroutine_threadsafe(send_queue.put({"type": "sources", "session_id": session_id, "items": sources}), loop)
                            asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_final", "text": answer, "session_id": session_id}), loop)
                            try:
                                memory.append({"role": "assistant", "content": answer})
                                if len(memory) > 16:
                                    del memory[0:len(memory)-16]
                            except Exception:
                                pass
                            return
                    except Exception:
                        pass
                prompt = "You are a helpful assistant. Answer the user's request clearly, concisely, and conversationally.\n"
                try:
                    if memory:
                        prompt += "Conversation so far (most recent last):\n"
                        for m in memory[-8:]:
                            prompt += f"{m.get('role','user').title()}: {m.get('content','')}\n"
                    prompt += f"User: {text}\nRespond directly to the user."
                except Exception:
                    prompt += f"User: {text}\nRespond directly to the user."
                model = genai.GenerativeModel('gemini-1.5-flash')
                stream = model.generate_content(prompt, stream=True)
                started = False
                accum_parts = []
                for chunk in stream:
                    part = getattr(chunk, 'text', '') or ''
                    if part:
                        accum_parts.append(part)
                        if not started:
                            started = True
                            asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_stream_start", "session_id": session_id, "tts_unavailable": False if self.murf_key else True}), loop)
                        asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_delta", "text": part, "session_id": session_id}), loop)
                        if self.murf_key:
                            asyncio.run_coroutine_threadsafe(murf_queue.put({"text": part}), loop)
                try:
                    stream.resolve()
                except Exception:
                    pass
                full_text = ''.join(accum_parts).strip()
                asyncio.run_coroutine_threadsafe(send_queue.put({"type": "assistant_final", "text": full_text, "session_id": session_id}), loop)
                try:
                    if full_text:
                        memory.append({"role": "assistant", "content": full_text})
                        if len(memory) > 16:
                            del memory[0:len(memory)-16]
                except Exception:
                    pass
                if self.murf_key:
                    asyncio.run_coroutine_threadsafe(murf_queue.put({"end": True}), loop)
            except Exception as e:
                logger.exception(f"[TurnDetection] LLM streaming error: {e}")

        def on_final(text: str):
            nonlocal murf_future
            static_ctx = f"{session_id}_ctx"
            if not murf_future or murf_future.done():
                try:
                    murf_future = asyncio.run_coroutine_threadsafe(murf_ws_worker(static_ctx), loop)
                except RuntimeError as e:
                    logger.warning(f"[TurnDetection] Could not schedule Murf WS worker: {e}")
            threading.Thread(target=stream_llm_response, args=(text,), daemon=True).start()

        state = {"latest_partial": None, "got_final": False}
        self.wire_handlers(client, loop, session_id, send_queue, on_final, state)
        self.connect(client)

        await websocket.send_text(json.dumps({"type": "connection_established", "message": "Turn detection ready", "session_id": session_id}))

        async def recv_audio():
            nonlocal paused, use_search
            while True:
                try:
                    msg = await websocket.receive()
                except WebSocketDisconnect:
                    try:
                        if not state.get("got_final") and state.get("latest_partial"):
                            final_text = state["latest_partial"]
                            try:
                                add_transcription_to_cache(final_text, session_id)
                            except Exception:
                                pass
                            await send_queue.put({"type": "final_transcript", "text": final_text, "session_id": session_id, "turn_order": 1, "is_formatted": False})
                            await send_queue.put({"type": "turn_end", "session_id": session_id, "turn_order": 1})
                            on_final(final_text)
                    except Exception:
                        pass
                    break
                except RuntimeError:
                    break
                if "bytes" in msg:
                    if not paused:
                        client.stream(msg["bytes"])
                elif "text" in msg and msg["text"] == "stop_streaming":
                    try:
                        client.disconnect(terminate=True)
                    except Exception:
                        pass
                    try:
                        if not state.get("got_final") and state.get("latest_partial"):
                            final_text = state["latest_partial"]
                            try:
                                add_transcription_to_cache(final_text, session_id)
                            except Exception:
                                pass
                            await send_queue.put({"type": "final_transcript", "text": final_text, "session_id": session_id, "turn_order": 1, "is_formatted": False})
                            await send_queue.put({"type": "turn_end", "session_id": session_id, "turn_order": 1})
                            on_final(final_text)
                    except Exception:
                        pass
                    break
                elif "text" in msg and msg["text"] in ("pause", "pause_streaming"):
                    paused = True
                    await send_queue.put({"type": "paused", "session_id": session_id})
                elif "text" in msg and msg["text"] in ("resume", "resume_streaming"):
                    paused = False
                    await send_queue.put({"type": "resumed", "session_id": session_id})
                elif "text" in msg and msg["text"] in ("search_on", "search_off"):
                    use_search = (msg["text"] == "search_on")
                    await send_queue.put({"type": "mode", "search": use_search, "session_id": session_id})

        async def send_messages():
            while True:
                try:
                    payload = await asyncio.wait_for(send_queue.get(), timeout=2.0)
                    try:
                        await websocket.send_text(json.dumps(payload))
                    except Exception:
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
            if murf_future and not murf_future.done():
                with contextlib.suppress(Exception):
                    await asyncio.wrap_future(murf_future)
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info("[TurnDetection] WebSocket closed")
