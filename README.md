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

🎙️ AI Voice Studio

AI Voice Studio is a multilingual speech AI application for working with text, speech, and audio in Uzbek, English, and Korean.

The project is designed as a practical portfolio project demonstrating how open-source speech models can be integrated into a modern, responsive AI application and deployed with Hugging Face Spaces.

✨ Features

📝 Text → Speech

Convert text into natural-sounding speech.

Supported languages:

🇺🇿 Uzbek

🇬🇧 English

🇰🇷 Korean

🎤 Speech → Text

Transcribe uploaded or recorded audio using a multilingual speech recognition model.

Supported languages:

🇺🇿 Uzbek

🇬🇧 English

🇰🇷 Korean

🔊 Audio Enhancement

Improve recorded audio with:

Noise reduction

Silence trimming

Audio normalization

👤 Voice Cloning

A dedicated voice-cloning interface is included in the application architecture.

Voice cloning model integration is planned for the next development stage. The current version does not claim active voice cloning functionality.

🌍 Multilingual Interface

The entire user interface is localized for:

🇺🇿 O'zbek

🇬🇧 English

🇰🇷 한국어

Changing the interface language also updates navigation labels, buttons, headings, descriptions, and other visible UI text.

🧠 Models

The current implementation uses open-source Hugging Face models:

Text → Speech

facebook/mms-tts-uzb-script_cyrillic

facebook/mms-tts-eng

facebook/mms-tts-kor

The Uzbek MMS model uses Cyrillic input, so the application includes a lightweight Latin-to-Cyrillic conversion step for Uzbek text.

Speech → Text

openai/whisper-large-v3-turbo

Models are loaded lazily to reduce unnecessary memory usage and improve application startup.

🛠️ Tech Stack

Python 3.10

Gradio 6

PyTorch

Transformers

Hugging Face Spaces

Hugging Face Hub

Whisper

Facebook MMS TTS

Librosa

SoundFile

Noisereduce

SciPy

NumPy

Spaces SDK / ZeroGPU-compatible architecture

🏗️ Architecture

                    ┌─────────────────────┐
                    │   AI Voice Studio   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       Text → Speech     Speech → Text    Audio Enhancement
              │                │                │
              ▼                ▼                ▼
          MMS TTS           Whisper       Librosa +
       UZ / EN / KO         Multilingual   Noisereduce
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                         Audio / Text

🎨 UI / UX

The interface follows a clean green-and-white visual system with:

Responsive layout

Animated navigation

Multilingual navigation labels

Clear feature panels

Accessible controls

Modern cards and visual hierarchy

Developer/portfolio section

Mobile-friendly styling

The application is intended to be easy to demonstrate during technical interviews and portfolio reviews.

⚡ Deployment

The application is designed for deployment on Hugging Face Spaces.

The inference functions use the Spaces GPU decorator so the same application architecture can be adapted to ZeroGPU or other GPU hardware when available.

The code also includes CPU-compatible execution logic for environments where CUDA is unavailable.

📁 Project Structure

AI-voice-Studio/
│
├── app.py
├── README.md
├── requirements.txt
├── logo.png
└── portfoliyo.jpg

Assets

logo.png — AI Voice Studio branding/logo

portfoliyo.jpg — developer profile image

🚀 Run Locally

Clone the repository:

git clone https://huggingface.co/spaces/IKROMJON01/AI-voice-Studio
cd AI-voice-Studio

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

Then open the local Gradio interface shown in the terminal.

🔒 Responsible AI

Speech technologies can be powerful and should be used responsibly.

For voice cloning, use only:

Your own voice

Voice samples for which you have explicit permission

Do not use the application to impersonate another person without authorization.

📌 Current Status

Feature

Status

Text → Speech

✅ Available

Uzbek TTS

✅ Available

English TTS

✅ Available

Korean TTS

✅ Available

Speech → Text

✅ Available

Uzbek STT

✅ Available

English STT

✅ Available

Korean STT

✅ Available

Audio Enhancement

✅ Available

Multilingual UI

✅ Available

Responsive UI

✅ Available

Voice Cloning UI

🧩 Architecture ready

Voice Cloning Model

🚧 Next stage

🎯 Portfolio Goal

AI Voice Studio demonstrates practical experience in:

Speech AI

Natural Language Processing

Multilingual AI

Text-to-Speech

Automatic Speech Recognition

Audio preprocessing

Model integration

GPU/CPU inference

Hugging Face deployment

Gradio application development

Responsible AI design

👨‍💻 Developer

TOJIBOEV IKROMJON MAHKHAMBOY UGLI

AI/ML Engineer focused on practical AI applications, Speech AI, Computer Vision, and open-source machine learning models.

📧 ikromjonkorealife@gmail.com

AI Voice Studio · Open-source Speech AI · UZ / EN / KO
