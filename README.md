---
title: AI Voice Studio
emoji: 🌱
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: "6.10.0"
app_file: app.py
pinned: false
---

# 🌱 AI Voice Studio

A multilingual AI speech studio for **English, Korean, and Uzbek**, built for practical speech-AI experiments and portfolio demonstration.

## Features

- 📝 Text → Speech
  - 🇺🇿 Uzbek
  - 🇬🇧 English
  - 🇰🇷 Korean
- 🎤 Speech → Text
  - 🇺🇿 Uzbek
  - 🇬🇧 English
  - 🇰🇷 Korean
- 🔊 Audio Enhancement
  - Noise reduction
  - Silence trimming
  - Peak normalization
- 👤 Voice Cloning UI prepared for the next multilingual model integration
- 🌍 Fully localized interface: Uzbek / English / Korean
- ✨ Animated, responsive green-and-white UI

## Models

### Speech → Text
`openai/whisper-large-v3-turbo`

### Text → Speech
- `facebook/mms-tts-uzb-script_cyrillic`
- `facebook/mms-tts-eng`
- `facebook/mms-tts-kor`

## ZeroGPU / CPU compatibility

This Space uses Hugging Face **ZeroGPU** for GPU-dependent inference. TTS and STT functions are decorated with `@spaces.GPU`, so GPU resources are requested only while inference is running.

The same code can fall back to CPU when `AI_VOICE_DEVICE=cpu` is set, making it easier to test on a regular CPU Space.

Models are loaded lazily to avoid loading all speech models during application startup.

## Responsible AI

Use voice cloning only with your own voice or with explicit permission from the speaker.

## Model licenses

Each model retains its own license and usage restrictions. Check the individual model card before commercial use.
