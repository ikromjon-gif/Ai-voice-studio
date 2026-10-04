# 🎙️ AI Voice Studio

> **Multilingual Speech AI for TTS, ASR, and practical audio processing.**

AI Voice Studio supports **Uzbek, English, and Korean** and demonstrates open-source speech-model integration, resource-aware inference, multilingual UI, and Hugging Face deployment.

**Live demo:** https://huggingface.co/spaces/IKROMJON01/AI-voice-Studio

## Capabilities

### Text → Speech
- Uzbek, English, Korean
- Hugging Face MMS TTS models
- Lazy model loading

### Speech → Text
- Whisper Large V3 Turbo
- Uzbek / English / Korean
- Upload or microphone input

### Audio Enhancement
- Noise reduction
- Silence trimming
- Peak normalization
- WAV export

### Voice Cloning
The public interface contains the cloning architecture, while the cloning model remains disabled until a suitable multilingual model is integrated and evaluated.

Only use voice cloning with the speaker's permission.

## Models

| Capability | Model |
|---|---|
| Uzbek TTS | facebook/mms-tts-uzb-script_cyrillic |
| English TTS | facebook/mms-tts-eng |
| Korean TTS | facebook/mms-tts-kor |
| Multilingual ASR | openai/whisper-large-v3-turbo |

Models are loaded lazily to reduce unnecessary CPU/RAM/VRAM use.

## Architecture

```
AI Voice Studio
       |
 +-----+---------+
 |       |       |
 v       v       v
TTS     ASR    Audio Processing
 |       |       |
 +-------+-------+
         |
         v
    Gradio App
         |
         v
 Hugging Face Spaces
```

## Stack

Python · PyTorch · Transformers · Gradio · Hugging Face Spaces · Whisper · Facebook MMS TTS · Librosa · SoundFile · SciPy

## Engineering Highlights

- Lazy loading for large speech models
- CPU/GPU-compatible inference paths
- Multilingual model routing
- Lightweight audio preprocessing
- Hugging Face deployment
- Interactive Gradio UI

## Local Run

```bash
git clone https://github.com/ikromjon-gif/Ai-voice-studio.git
cd Ai-voice-studio
pip install -r requirements.txt
python app.py
```

## Responsible AI

- Use voice cloning only with your own voice or explicit permission.
- Do not use generated audio for unauthorized impersonation.
- Treat synthetic speech as generated content.

## Status

| Feature | Status |
|---|---|
| Uzbek TTS | Available |
| English TTS | Available |
| Korean TTS | Available |
| Uzbek STT | Available |
| English STT | Available |
| Korean STT | Available |
| Audio Enhancement | Available |
| Multilingual UI | Available |
| Voice Cloning Model | Future |

## Portfolio Signal

Speech AI · TTS / ASR · Transformers · Multilingual AI · Audio Processing · Resource-aware inference · Model integration · Hugging Face deployment

**Author:** Ikromjon Tojiboev