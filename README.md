# 🎙️ AI Voice Studio

> **Multilingual Speech AI for Text-to-Speech, Speech-to-Text, and Audio Enhancement.**

AI Voice Studio is a practical multilingual Speech AI application built with open-source models and deployed on Hugging Face Spaces. It supports **Uzbek, English, and Korean** and is designed as a portfolio project demonstrating model integration, inference optimization, multilingual UI/UX, and GPU/CPU-ready deployment.

## 🚀 Live Demo

👉 **[Try AI Voice Studio on Hugging Face Spaces](https://huggingface.co/spaces/IKROMJON01/AI-voice-Studio)**

## ✨ Features

### 📝 Text → Speech

Convert text into speech with language-specific Hugging Face MMS TTS models.

- 🇺🇿 Uzbek
- 🇬🇧 English
- 🇰🇷 Korean
- Lazy model loading to reduce startup memory usage
- Uzbek Latin-to-Cyrillic preprocessing for the selected MMS checkpoint

### 🎤 Speech → Text

Transcribe uploaded or recorded speech using **Whisper Large V3 Turbo**.

- 🇺🇿 Uzbek
- 🇬🇧 English
- 🇰🇷 Korean
- Upload or microphone input
- Language-aware transcription

### 🔊 Audio Enhancement

A lightweight audio preprocessing pipeline for improving recorded speech:

- Noise reduction
- Silence trimming
- Peak normalization
- WAV output

### 👤 Voice Cloning

The application includes a dedicated voice-cloning interface and responsible-use guidance. The current public version keeps the cloning model disabled until a suitable multilingual cloning model is integrated and tested for the target languages.

> **Responsible use:** only use your own voice or voice samples for which you have explicit permission. Do not use voice cloning for unauthorized impersonation.

## 🌍 Multilingual Interface

The **entire interface** can be switched between:

- 🇺🇿 **O'zbek**
- 🇬🇧 **English**
- 🇰🇷 **한국어**

Changing the interface language updates navigation, headings, labels, buttons, descriptions, developer information, and other visible UI text.

The feature-level language selectors independently provide the three supported languages for TTS, STT, and the voice-cloning interface.

## 🧠 Models

### Text → Speech

| Language | Model |
|---|---|
| 🇺🇿 Uzbek | `facebook/mms-tts-uzb-script_cyrillic` |
| 🇬🇧 English | `facebook/mms-tts-eng` |
| 🇰🇷 Korean | `facebook/mms-tts-kor` |

### Speech → Text

| Task | Model |
|---|---|
| Multilingual ASR | `openai/whisper-large-v3-turbo` |

Models are loaded **lazily**, meaning a model is loaded only when its corresponding feature is used. This helps reduce unnecessary CPU/RAM/VRAM usage in constrained deployment environments.

## 🏗️ Architecture

```text
                         AI Voice Studio
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
       Text → Speech      Speech → Text      Audio Enhancement
             │                  │                  │
             ▼                  ▼                  ▼
        Facebook MMS         Whisper       Librosa + Noisereduce
         UZ / EN / KO      Large V3 Turbo        + SciPy
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                         Audio / Text Output
```

## 🛠️ Tech Stack

- Python 3.10
- PyTorch
- Transformers
- Gradio
- Hugging Face Spaces
- Hugging Face Spaces SDK
- Whisper
- Facebook MMS TTS
- Librosa
- SoundFile
- Noisereduce
- SciPy
- NumPy

## ⚡ Deployment

The project is structured for **Hugging Face Spaces** and uses `@spaces.GPU` for inference functions so the same application can run with ZeroGPU-style dynamic GPU allocation when available.

The application also contains CPU-compatible inference logic and can be adapted to normal CPU or NVIDIA GPU hardware.

### Current deployment configuration

```text
Hugging Face Spaces
        │
        ▼
   Gradio App
        │
   ┌────┴────┐
   │         │
 CPU path   GPU path
   │         │
   └────┬────┘
        ▼
 Speech AI Models
```

## 📁 Project Structure

```text
Ai-voice-studio/
├── app.py
├── README.md
├── requirements.txt
├── logo.png
└── portfoliyo.jpg
```

- `app.py` — Gradio application, model loading, inference, multilingual UI, and styling
- `requirements.txt` — Python dependencies
- `logo.png` — AI Voice Studio branding
- `portfoliyo.jpg` — developer profile image

## 💻 Run Locally

```bash
git clone https://github.com/ikromjon-gif/Ai-voice-studio.git
cd Ai-voice-studio
pip install -r requirements.txt
python app.py
```

The application will start with the local Gradio interface shown in the terminal.

## 🔒 Responsible AI

AI-generated speech and voice technologies can be misused. This project follows a responsible-use approach:

- Use voice cloning only with your own voice or explicit permission.
- Do not impersonate another person without authorization.
- Treat generated speech as synthetic content.
- Do not use model output as a substitute for professional or safety-critical decisions.

## 📌 Current Status

| Feature | Status |
|---|---|
| 📝 Uzbek TTS | ✅ Available |
| 📝 English TTS | ✅ Available |
| 📝 Korean TTS | ✅ Available |
| 🎤 Uzbek STT | ✅ Available |
| 🎤 English STT | ✅ Available |
| 🎤 Korean STT | ✅ Available |
| 🔊 Audio Enhancement | ✅ Available |
| 🌍 Uzbek / English / Korean UI | ✅ Available |
| 📱 Responsive UI | ✅ Available |
| 👤 Voice Cloning UI | 🧩 Architecture ready |
| 👤 Voice Cloning model | 🚧 Next stage |

## 🎯 Portfolio Focus

AI Voice Studio demonstrates practical experience in:

- Speech AI
- Text-to-Speech (TTS)
- Automatic Speech Recognition (ASR)
- Multilingual AI
- Audio preprocessing
- Transformer-based model integration
- Lazy model loading
- CPU/GPU inference
- Hugging Face deployment
- Gradio UI/UX development
- Responsible AI design

## 🔬 Project Direction

Future development may include:

- High-quality multilingual voice cloning
- Better Korean and English voice quality
- Additional Uzbek speech models
- Speaker/style controls
- Speech segmentation and timestamps
- More advanced audio restoration
- API deployment for external applications

## 👨‍💻 Developer

**TOJIBOEV IKROMJON MAHKHAMBOY UGLI**  
AI/ML Engineer · Computer Engineering · Chonnam National University

📧 **ikromjonkorealife@gmail.com**

Focused on practical AI applications, Speech AI, Computer Vision, and open-source machine learning models.

---

⭐ **AI Voice Studio** — Open-source Speech AI for **UZ / EN / KO**.
