# BuddyBot – 30 Days of Building a Real‑Time Voice Agent

BuddyBot is a real‑time voice agent I designed and improved over 30 days as part of the Murf AI 30 Days of Voice Agents challenge. I iterated daily, wired multiple APIs together, refined streaming behavior to work with partial data chunks, and focused on making conversations feel responsive and natural. The assistant listens to you, understands in real time, thinks using an LLM, and speaks back using high‑quality TTS. It also maintains short‑term memory to keep conversations coherent.

Live demo: https://tinyurl.com/buddy-bot

## What I built in these 30 days

- Real‑time speech to text using streaming, so partial results appear quickly instead of waiting until the end.
- A conversational loop where transcripts feed the LLM and the LLM response is streamed to speech.
- Lightweight context memory so follow‑ups like “and what about tomorrow” make sense.
- Focus on practical skills such as weather and stock price lookups.
- A simple web client that shows status and streams audio.

## How it works

1. Your browser captures audio and sends it to the backend over WebSocket.
2. AssemblyAI Universal Streaming transcribes speech as you talk and emits partial and final transcripts.
3. The transcript goes to Google Gemini for a concise, helpful response.
4. The response is synthesized to speech using Murf AI and streamed back for smooth playback.
5. A small chat memory keeps the last messages so follow‑ups feel natural.

## APIs and keys

Required
- ASSEMBLYAI_API_KEY: for streaming speech recognition.
- GEMINI_API_KEY: for language model responses (Google Generative AI / Gemini).

Recommended
- MURF_API_KEY: for text‑to‑speech.

Optional skills
- OPENWEATHER_API_KEY: for weather.
- TAVILY_API_KEY: for web search summaries.

You can set these keys via a local .env file during development or as environment variables on your deployment platform. This repository includes an `.env.example` with the expected names.

## Run locally

Prerequisites
- Python 3.11

Install dependencies
```bash
# From the project root folder
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Configure environment
```bash
# Windows PowerShell
copy .env.example .env

# Then edit .env and add your keys
# ASSEMBLYAI_API_KEY=...
# GEMINI_API_KEY=...
# MURF_API_KEY=...
# OPENWEATHER_API_KEY=... (optional)
# TAVILY_API_KEY=... (optional)
```

Start the server
```bash
python main.py
```

Open the UI at http://localhost:8080 and allow microphone access in your browser. Use the big record button to start and stop. The app will display transcripts and play generated speech when keys are configured.

## Key endpoints

- GET /              serves the web UI
- GET /health        reports if STT, LLM, and TTS are configured
- WS  /ws/streaming  real‑time transcription with turn detection
- WS  /ws/turn-detection  focused turn detection endpoint
- POST /transcribe/file   transcribe an uploaded file
- POST /agent/chat/{session_id}   full pipeline for file input
- GET  /agent/history/{session_id}   get recent messages
- DELETE /agent/history/{session_id} clear recent messages
- POST /generate-audio   synthesize text to speech

## Deploying on Render

This repository includes a `render.yaml` so you can deploy with the Blueprint flow. The service binds to `$PORT` and exposes `/health` for checks. If you prefer a manual Web Service, use the following settings:

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`
- Python: 3.11

Set the environment variables listed above in the service settings and redeploy. After it turns live, verify `/health` and the homepage.

## Notes and troubleshooting

- If you do not hear audio, confirm that MURF_API_KEY is set, and your browser tab has microphone permission.
- If transcripts do not appear, check ASSEMBLYAI_API_KEY and the WebSocket connection in the browser console.
- If LLM responses are empty, verify GEMINI_API_KEY and review logs for rate limits or model errors.
- On slow networks, initial responses can take a few seconds because multiple services are involved.

## About the challenge

This project was built and refined day by day during the 30 Days of Voice Agents challenge by Murf AI. The goal was to learn by shipping small improvements daily: wiring APIs, shaping prompts, handling streaming chunks, improving error handling, and making the interaction feel natural. The result is a compact, practical voice agent you can run locally or host in the cloud.
