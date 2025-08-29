# BuddyBot - A 30-Day Journey with Murf AI

Hey there! Let me tell you about this cool voice assistant I built. It's like having a conversation with a really smart friend who never gets tired of talking to you. You speak to it, it thinks about what you said, and then talks back to you with a natural voice.

## What Does This Thing Actually Do

So imagine this - you're sitting at your computer and you just want to have a chat with someone who's really smart. You click a button, say whatever's on your mind, and this AI actually understands you and responds back. Not just with boring text, but with actual speech that sounds pretty human.

The coolest part? It remembers what you talked about before. So if you mentioned your dog earlier, it might ask about your dog later in the conversation. It's like having a conversation with someone who actually listens and cares about what you're saying.

Basically, I wanted to build something that felt more like talking to a person than using a computer program.

## The Cool Stuff It Can Do

This thing is packed with features that actually make sense:

You can talk to it naturally and it understands what you're saying
It thinks about your words and gives you smart responses back
It talks back to you with a voice that doesn't sound like a robot
It remembers everything you talked about in your conversation
The website looks pretty nice and shows you when it's listening or thinking
You can use it through a simple web page or hook into it with code
When something breaks it doesn't just crash - it tries to help you figure out what went wrong
You can have different conversations and it keeps them separate

The whole point was to make something that felt easy and natural to use, not like you're operating some complicated machine.

## The Tech Stuff (Don't Worry, I'll Keep It Simple)

I used a bunch of different technologies to make this work, but here's what they do in plain English:

FastAPI handles all the behind-the-scenes server work (it's like the brain of the operation)
Regular HTML and CSS make the website look good
JavaScript makes the website actually do stuff when you click buttons
AssemblyAI figures out what you said when you talk to it
Google Gemini is the smart part that thinks of good responses
Murf AI makes the voice that talks back to you sound natural
Your web browser handles recording your voice

I picked these because they work well together and I could actually understand how to use them.

## How I Built This Thing

I kept it pretty straightforward with three main parts:

The website you see - This is the pretty part that you interact with
The server that does the work - This is where all the magic happens
The connections to AI services - This is how it gets smart and learns to talk

Here's what happens when you use it (it's actually pretty cool):

1. You click the record button and start talking
2. Your browser grabs the audio and sends it to my server
3. AssemblyAI listens to your audio and figures out what words you said
4. Google Gemini reads those words and thinks of something smart to say back
5. Murf AI takes that smart response and turns it into speech that sounds human
6. Your browser plays the response and shows you the conversation

The whole thing happens pretty fast, which honestly still amazes me sometimes.

## Want to Try It Out Yourself?

Great! Here's how to get it running on your computer. Don't worry, I'll walk you through it step by step.

### What You Need First

You just need Python installed. If you don't have it, go grab version 3.7 or newer from python.org.

### Getting Your API Keys

This is the slightly annoying part - you need to sign up for three services to get API keys. Think of these as permission slips that let my code use their AI services:

Go to AssemblyAI and sign up - they handle the speech recognition part
Head over to Google AI Studio and get a Gemini key - this is the smart conversation part
Sign up at Murf AI for their API key - this makes the voice responses sound good

Yeah, it's a bit of work, but these services are what make the magic happen.

### Setting Everything Up

1. Download all my code to a folder on your computer

2. Open up your terminal or command prompt and go to that folder

3. Install all the Python stuff it needs:
```bash
pip install -r requirements.txt
```

4. Copy the `.env.example` file to `.env` and add your API keys:
```bash
cp .env.example .env
```

Then edit the `.env` file and replace the placeholder values with your actual API keys:
```
ASSEMBLYAI_API_KEY=your_actual_assemblyai_api_key
GEMINI_API_KEY=your_actual_gemini_api_key 
MURF_API_KEY=your_actual_murf_api_key
```

5. Start it up:
```bash
python main.py
```

6. Open your web browser and go to: http://localhost:8080

That's it! If everything worked, you should see a nice looking page with a big record button.

### How to Actually Use This Thing

Using it is the fun part and super easy:

1. Click that big "Start Recording" button
2. Talk to it like you're talking to a friend - ask questions, tell jokes, whatever
3. Click stop when you're done talking  
4. Wait a few seconds while it thinks about what you said
5. Listen to it talk back to you
6. Keep the conversation going as long as you want

The AI will remember what you talked about, so you can reference things from earlier in your conversation. It's pretty neat how it keeps track of context.

## API Endpoints

The application provides a comprehensive REST API:

**GET /**: Main web interface
**POST /transcribe/file**: Upload audio file for transcription  
**POST /agent/chat/{session_id}**: Full conversational pipeline with session memory
**GET /agent/history/{session_id}**: Retrieve conversation history
**DELETE /agent/history/{session_id}**: Clear conversation history
**POST /generate-audio**: Convert text to speech
**POST /llm/query**: Query AI with audio input
**GET /health**: Check system status and API availability

## Deploying to Render (Free Tier)

Fast path:

- Push this repo to GitHub
- In Render, create a new Web Service from your repo
- When asked, set:
    - Build Command: `pip install -r requirements.txt`
    - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
    - Runtime: Python 3.11
- Add Environment Variables (from your `.env`): `ASSEMBLYAI_API_KEY`, `GEMINI_API_KEY`, `MURF_API_KEY`, `OPENWEATHER_API_KEY`, `TAVILY_API_KEY`

This repo includes a `render.yaml` so you can also click “New +” → “Blueprint” and point to the repo; Render will auto-provision using the settings above. Health check path is `/health`.

Notes:

- Public URL will be HTTPS; the app serves the UI at `/` and static files at `/static`
- WebSockets endpoints used by the UI:
    - `/ws/turn-detection` (primary)
    - `/ws/streaming` (advanced, optional)
- Ensure your browser allows microphone access when visiting the Render URL

## Project Structure

```
Task-13/
├── main.py              # FastAPI backend server
├── index.html           # Main web interface  
├── requirements.txt     # Python dependencies
├── .env                 # API keys (create this file)
├── static/
│   ├── styles.css       # Modern UI styling
│   ├── script.js        # Frontend JavaScript logic
│   └── logo.jpeg        # Murf AI logo
└── uploads/
    └── recording.wav    # Temporary audio storage
```

## My 30-Day Adventure Building This

This project was part of the Murf AI 30-Day Challenge, and honestly, it was quite a journey. I started with basically no idea how to make computers understand speech, and somehow ended up with this working voice assistant.

Here's how it evolved day by day:

First, I just wanted to make text turn into speech - seemed simple enough, right?
Then I thought "what if it could understand when I talk to it too?"
Next came the idea of making it actually smart and able to hold conversations
I spent way too much time making the website look nice because first impressions matter
Added the ability to remember conversations because nobody likes repeating themselves
Put in lots of error handling because things break and users get frustrated
Finally turned it into a proper API so other people could build on top of it

Each day I learned something new, broke something that was working, fixed it, and then broke something else. That's just how coding goes, I guess.

What I'm most proud of is that it actually works reliably. You can have real conversations with it, and it doesn't feel like you're talking to a computer most of the time.

## When Things Go Wrong (And They Will)

Here are the most common issues I ran into while building this, and what usually fixes them:

**Can't record audio**: Your browser is probably blocking microphone access. Look for a little microphone icon in the address bar and click "allow"

**Getting API errors**: Double-check that you copied your API keys correctly into the .env file. No extra spaces or quotes where they shouldn't be

**Server won't start**: Make sure you installed all the Python requirements. Sometimes running `pip install -r requirements.txt` again fixes weird issues

**It can't understand what I'm saying**: Try speaking a bit slower and closer to your microphone. Also, make sure you're in a quiet room

**Everything is super slow**: Yeah, that's normal. The AI has to process your speech through three different services, so it takes a few seconds. Grab a coffee while you wait

**The voice sounds weird**: That's just how AI voices are sometimes. Murf AI is pretty good, but it's not perfect

Most problems come down to either microphone permissions or API key issues. When in doubt, check those first.

## Cool Ideas for Making This Even Better

I've got a bunch of ideas for where this could go next:

Make it work in different languages so more people can use it
Add voice cloning so it could sound like specific people (with their permission, of course)
Build analytics so you can see patterns in your conversations
Create a mobile app version because not everyone wants to use a computer
Make the responses faster by streaming instead of waiting for everything to process
Add user accounts so people can save their conversation history long-term

If you want to mess around with any of these ideas, feel free to take my code and run with it. That's how we all learn and build cool stuff.

## Why I Built This

Honestly, I just thought it would be cool to have a conversation with an AI that felt natural. Most voice assistants feel very robotic and limited - you can ask them the weather or set a timer, but you can't really chat with them.

I wanted to build something that felt more like talking to a smart friend who happens to live in your computer. Someone you could bounce ideas off of, ask random questions, or just have a conversation with when you're bored.

The technical challenge was fun too. Connecting all these different AI services together and making them work smoothly was like solving a complex puzzle. Each piece had to fit just right or the whole thing would fall apart.

But mostly, I built this because I believe voice interfaces are the future. Typing is fine, but talking is more natural for humans. We've been doing it for thousands of years, after all.

---

This project taught me a ton about AI, web development, and how to make different technologies work together. Every bug I fixed and every feature I added helped me understand something new about how computers can understand and respond to human speech.

The goal wasn't just to build something that worked - it was to build something that people would actually want to use and talk to. I think I got pretty close to that goal.
