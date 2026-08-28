import os
import re
import base64
from pathlib import Path

# Import spaces before torch: ZeroGPU installs its CUDA emulation hooks here.
import spaces
import tempfile

import gradio as gr
import torch
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    VitsModel,
)

APP_TITLE = "AI Voice Studio"

# CPU Basic compatible.
# If you later move this Space to a normal NVIDIA GPU, set:
# AI_VOICE_DEVICE=cuda
DEVICE_MODE = os.getenv("AI_VOICE_DEVICE", "cuda" if os.getenv("SPACE_ID") else "cpu").lower()
DEVICE = "cuda" if DEVICE_MODE == "cuda" else "cpu"

# In ZeroGPU, CUDA is available only while a @spaces.GPU function is running.
# Keep the default CPU-safe, then select the runtime device inside the decorated
# inference functions. This also keeps the same code usable on CPU Basic.
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def inference_device():
    if DEVICE_MODE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def inference_dtype(device):
    return torch.float16 if device == "cuda" else torch.float32

WHISPER_ID = "openai/whisper-large-v3-turbo"

TTS_MODELS = {
    "uz": "facebook/mms-tts-uzb-script_cyrillic",
    "en": "facebook/mms-tts-eng",
    "ko": "facebook/mms-tts-kor",
}

LANGUAGES = {
    "uz": {"name": "🇺🇿 O'zbek", "whisper": "uz"},
    "en": {"name": "🇬🇧 English", "whisper": "en"},
    "ko": {"name": "🇰🇷 한국어", "whisper": "ko"},
}

print(f"[AI Voice Studio] device={DEVICE}")

# ---------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------
# CPU Basic has limited resources. Models are loaded only when needed.
# This avoids loading three TTS models + Whisper at startup.
# ---------------------------------------------------------------------

_whisper_processor = None
_whisper_model = None
_tts_cache = {}


def get_whisper(device):
    global _whisper_processor, _whisper_model

    if _whisper_processor is None or _whisper_model is None:
        print(f"[AI Voice Studio] Loading Whisper on {device}...")
        _whisper_processor = AutoProcessor.from_pretrained(WHISPER_ID)
        _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_ID,
            torch_dtype=inference_dtype(device),
        ).to(device)
        _whisper_model.eval()

    elif str(next(_whisper_model.parameters()).device) != device:
        _whisper_model = _whisper_model.to(device)

    return _whisper_processor, _whisper_model


def get_tts(language, device):
    if language not in TTS_MODELS:
        raise ValueError(f"Unsupported TTS language: {language}")

    cache_key = f"{language}:{device}"
    if cache_key not in _tts_cache:
        model_id = TTS_MODELS[language]
        print(f"[AI Voice Studio] Loading TTS: {model_id} on {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(
            model_id,
            torch_dtype=inference_dtype(device),
        ).to(device)
        model.eval()

        _tts_cache[cache_key] = (tokenizer, model)

    return _tts_cache[cache_key]


def clean_text(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def latin_to_cyrillic(text):
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("ʻ", "'")
    ).lower()

    for src, dst in [
        ("sh", "ш"), ("ch", "ч"), ("ng", "нг"),
        ("g'", "ғ"), ("o'", "ў"),
        ("yo", "ё"), ("yu", "ю"), ("ya", "я"), ("ts", "ц"),
    ]:
        text = text.replace(src, dst)

    return text.translate(str.maketrans({
        "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф",
        "g": "г", "h": "ҳ", "i": "и", "j": "ж", "k": "к",
        "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
        "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у",
        "v": "в", "x": "х", "y": "й", "z": "з",
    }))


def save_wav(audio, sample_rate):
    audio = np.asarray(audio).squeeze()
    audio = np.nan_to_num(audio).astype(np.float32)

    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(path, np.clip(audio, -1, 1), int(sample_rate))
    return path


# ---------------------------------------------------------------------
# Text → Speech
# ---------------------------------------------------------------------

@spaces.GPU(duration=120)
def text_to_speech(text, language):
    text = clean_text(text)

    if not text:
        raise gr.Error("Please enter some text.")

    try:
        run_device = inference_device()
        tokenizer, model = get_tts(language, run_device)

        # Uzbek MMS checkpoint expects Cyrillic input.
        model_text = latin_to_cyrillic(text) if language == "uz" else text

        inputs = tokenizer(
            model_text,
            return_tensors="pt",
        ).to(run_device)

        with torch.inference_mode():
            output = model(**inputs)

        audio = output.waveform.squeeze().float().cpu().numpy()
        sample_rate = int(model.config.sampling_rate)

        return save_wav(audio, sample_rate)

    except Exception as exc:
        raise gr.Error(f"TTS failed: {exc}")


# ---------------------------------------------------------------------
# Speech → Text
# ---------------------------------------------------------------------

@spaces.GPU(duration=120)
def speech_to_text(audio_path, language):
    if not audio_path:
        raise gr.Error("Please upload or record an audio file.")

    try:
        audio, _ = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
        )

        if audio.size == 0:
            raise ValueError("The audio file is empty.")

        run_device = inference_device()
        processor, model = get_whisper(run_device)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            run_device,
            dtype=model.dtype,
        )

        language_code = LANGUAGES[language]["whisper"]

        with torch.inference_mode():
            predicted_ids = model.generate(
                input_features=input_features,
                language=language_code,
                task="transcribe",
                max_new_tokens=448,
            )

        return processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

    except Exception as exc:
        raise gr.Error(f"Transcription failed: {exc}")


# ---------------------------------------------------------------------
# Audio Enhancement
# ---------------------------------------------------------------------

def audio_fix(audio_path, denoise=True, trim_silence=True):
    if not audio_path:
        raise gr.Error("Please upload or record an audio file.")

    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=True,
        )

        if audio.size == 0:
            raise ValueError("The audio file is empty.")

        if trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=30)

        if denoise:
            audio = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=True,
            )

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = 0.95 * audio / peak

        return save_wav(audio, sample_rate)

    except Exception as exc:
        raise gr.Error(f"Audio enhancement failed: {exc}")


# ---------------------------------------------------------------------
# Multilingual UI
# ---------------------------------------------------------------------

I18N = {
    "uz": {
        "title": "AI Voice Studio",
        "subtitle": "Nutq, ovoz va audio bilan ishlash uchun AI studiyasi.",
        "tts": "📝 Matn → Ovoz",
        "stt": "🎤 Ovoz → Matn",
        "clone": "👤 Ovoz klonlash",
        "fix": "🔊 Audio yaxshilash",
        "language": "Til",
        "text": "Matn",
        "reference": "Namuna ovoz",
        "generate": "Ovoz yaratish",
        "transcribe": "Matnga aylantirish",
        "enhance": "Audio yaxshilash",
        "output": "Natija audio",
        "transcript": "Transkripsiya",
        "denoise": "Shovqinni kamaytirish",
        "trim": "Jim qismlarni kesish",
        "tts_info": "O'zbek, English va Korean uchun Text → Speech mavjud.",
        "clone_info": "Ovoz klonlash keyingi model integratsiyasida yoqiladi.",
        "responsible": "⚠️ Ovoz klonlashni faqat o'zingizga tegishli yoki ruxsat berilgan ovozlar bilan ishlating.",
        "nav_tts": "📝 Matn → Ovoz",
        "nav_stt": "🎤 Ovoz → Matn",
        "nav_clone": "👤 Ovoz klonlash",
        "nav_fix": "🔊 Audio yaxshilash",
        "panel_tts": "Matnni tabiiy ovozga aylantiring",
        "panel_stt": "Ovozni tez va aniq matnga aylantiring",
        "panel_clone": "Ruxsat berilgan ovoz bilan ishlash",
        "panel_fix": "Audio sifatini yaxshilang",
        "audio": "Audio",
        "coming_next": "Ovoz klonlash modeli — keyingi bosqichda",
        "tts_placeholder": "Matn kiriting...",
        "clone_placeholder": "Ovoz klonlash uchun matn kiriting...",
        "badge_speech": "🎙️ Nutq AI",
        "badge_languages": "🌍 UZ · EN · KO",
        "badge_gpu": "⚡ ZeroGPU tayyor",
        "developer": "Dasturchi",
        "developer_bio": "AI/ML Engineer — amaliy AI, Speech AI, Computer Vision va open-source modellar ustida ishlaydi.",
        "footer": "🌱 AI Voice Studio · Open-source Speech AI · CPU/GPU qo'llab-quvvatlanadi",
    },
    "en": {
        "title": "AI Voice Studio",
        "subtitle": "An AI studio for speech, voice, and audio processing.",
        "tts": "📝 Text → Speech",
        "stt": "🎤 Speech → Text",
        "clone": "👤 Voice Cloning",
        "fix": "🔊 Audio Enhancement",
        "language": "Language",
        "text": "Text",
        "reference": "Reference Voice",
        "generate": "Generate Speech",
        "transcribe": "Transcribe",
        "enhance": "Enhance Audio",
        "output": "Generated Audio",
        "transcript": "Transcript",
        "denoise": "Reduce Noise",
        "trim": "Trim Silence",
        "tts_info": "Text → Speech is available for Uzbek, English, and Korean.",
        "clone_info": "Voice cloning will be enabled after the multilingual cloning model integration.",
        "responsible": "⚠️ Use voice cloning only with your own voice or with explicit permission.",
        "nav_tts": "📝 Text → Speech",
        "nav_stt": "🎤 Speech → Text",
        "nav_clone": "👤 Voice Cloning",
        "nav_fix": "🔊 Audio Enhancement",
        "panel_tts": "Turn text into natural speech",
        "panel_stt": "Convert speech into accurate text",
        "panel_clone": "Work with authorized voice samples",
        "panel_fix": "Improve and clean your audio",
        "audio": "Audio",
        "coming_next": "Voice cloning model — coming next",
        "tts_placeholder": "Enter text...",
        "clone_placeholder": "Enter the text for voice cloning...",
        "badge_speech": "🎙️ Speech AI",
        "badge_languages": "🌍 UZ · EN · KO",
        "badge_gpu": "⚡ ZeroGPU Ready",
        "developer": "Developer",
        "developer_bio": "AI/ML Engineer focused on practical AI, Speech AI, Computer Vision, and open-source models.",
        "footer": "🌱 AI Voice Studio · Open-source Speech AI · CPU/GPU Supported",
    },
    "ko": {
        "title": "AI Voice Studio",
        "subtitle": "음성, 목소리 및 오디오 처리를 위한 AI 스튜디오입니다.",
        "tts": "📝 텍스트 → 음성",
        "stt": "🎤 음성 → 텍스트",
        "clone": "👤 음성 클로닝",
        "fix": "🔊 오디오 개선",
        "language": "언어",
        "text": "텍스트",
        "reference": "참조 음성",
        "generate": "음성 생성",
        "transcribe": "텍스트 변환",
        "enhance": "오디오 개선",
        "output": "생성된 오디오",
        "transcript": "변환 결과",
        "denoise": "노이즈 감소",
        "trim": "무음 구간 제거",
        "tts_info": "우즈베크어, 영어 및 한국어 Text → Speech를 지원합니다.",
        "clone_info": "다국어 음성 클로닝 모델 통합 후 기능을 활성화할 예정입니다.",
        "responsible": "⚠️ 본인 소유의 음성 또는 명시적인 허가를 받은 음성만 사용하세요.",
        "nav_tts": "📝 텍스트 → 음성",
        "nav_stt": "🎤 음성 → 텍스트",
        "nav_clone": "👤 음성 클로닝",
        "nav_fix": "🔊 오디오 개선",
        "panel_tts": "텍스트를 자연스러운 음성으로 변환합니다",
        "panel_stt": "음성을 빠르고 정확하게 텍스트로 변환합니다",
        "panel_clone": "허가된 음성 샘플로 작업합니다",
        "panel_fix": "오디오 품질을 개선하고 정리합니다",
        "audio": "오디오",
        "coming_next": "음성 클로닝 모델 — 다음 단계에서 제공",
        "tts_placeholder": "텍스트를 입력하세요...",
        "clone_placeholder": "음성 클로닝을 위한 텍스트를 입력하세요...",
        "badge_speech": "🎙️ Speech AI",
        "badge_languages": "🌍 UZ · EN · KO",
        "badge_gpu": "⚡ ZeroGPU 지원",
        "developer": "개발자",
        "developer_bio": "AI/ML Engineer — 실용 AI, Speech AI, Computer Vision 및 오픈소스 모델을 개발합니다.",
        "footer": "🌱 AI Voice Studio · 오픈소스 Speech AI · CPU/GPU 지원",
    },
}


def local_image_data(filename, fallback=""):
    """Load a Space-local image as a data URI."""
    candidates = [
        Path(filename),
        Path(filename.lower()),
        Path(filename.upper()),
    ]
    # Also find the file case-insensitively in the Space root.
    target = Path(filename).name.lower()
    try:
        for item in Path(".").iterdir():
            if item.is_file() and item.name.lower() == target:
                candidates.insert(0, item)
    except Exception:
        pass

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            suffix = path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            continue
    return fallback


LOGO_DATA = local_image_data("logo.png")
PORTFOLIO_DATA = local_image_data("portfoliyo.jpg")

print(
    "[AI Voice Studio] assets: "
    f"logo={'OK' if LOGO_DATA else 'MISSING'}, "
    f"portfolio={'OK' if PORTFOLIO_DATA else 'MISSING'}"
)


def hero_html(language):
    t = I18N[language]
    logo = (
        f'<img src="{LOGO_DATA}" alt="AI Voice Studio logo" class="brand-logo">'
        if LOGO_DATA else
        '<div class="brand-mark">🎙️</div>'
    )
    return f"""
    <div class="hero">
        <div class="brand-mark">{logo}</div>
        <h1>{t["title"]}</h1>
        <p>{t["subtitle"]}</p>
        <div class="badges">
            <span>{t["badge_speech"]}</span>
            <span>{t["badge_languages"]}</span>
            <span>{t["badge_gpu"]}</span>
        </div>
    </div>
    """


CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background: #ffffff !important;
}
footer { display: none !important; }

.hero {
    text-align: center;
    padding: 30px 20px 22px;
    margin-bottom: 18px;
    border: 1px solid #dcfce7;
    border-radius: 24px;
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
    animation: heroIn .65s ease both;
}
.brand-mark {
    width: 58px;
    height: 58px;
    margin: 0 auto 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background: #16a34a;
    box-shadow: 0 10px 25px rgba(22,163,74,.18);
    font-size: 29px;
    animation: floatMark 3s ease-in-out infinite;
}
.brand-logo {
    width: 46px;
    height: 46px;
    object-fit: contain;
    display: block;
    border-radius: 12px;
}
.developer-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 18px;
    display: block;
}

.hero h1 {
    margin: 0;
    color: #14532d;
    font-size: 34px;
    font-weight: 800;
}
.hero p {
    max-width: 680px;
    margin: 8px auto 15px;
    color: #64748b;
    font-size: 15px;
}
.badges {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
}
.badges span {
    padding: 6px 11px;
    border-radius: 999px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    font-size: 12px;
    font-weight: 600;
}

/* Animated multilingual navigation */
.voice-nav {
    padding: 6px !important;
    margin: 4px 0 22px !important;
    border: 1px solid #dcfce7 !important;
    border-radius: 18px !important;
    background: #f8fffa !important;
    box-shadow: 0 8px 25px rgba(15,23,42,.05) !important;
}
.nav-btn {
    min-height: 50px !important;
    border-radius: 13px !important;
    border: 1px solid #dcfce7 !important;
    background: #ffffff !important;
    color: #166534 !important;
    font-weight: 700 !important;
    transition: transform .22s ease, box-shadow .22s ease, background .22s ease !important;
}
.nav-btn:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 9px 20px rgba(22,163,74,.15);
    border-color: #86efac !important;
}
.nav-btn:active {
    transform: scale(.98);
}
.nav-btn.active {
    background: #16a34a !important;
    color: white !important;
    border-color: #16a34a !important;
    box-shadow: 0 8px 18px rgba(22,163,74,.20);
}

.panel-card {
    padding: 22px !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    background: #ffffff !important;
    box-shadow: 0 10px 30px rgba(15,23,42,.06) !important;
    animation: panelIn .35s ease both;
}
.panel-title {
    margin: 0 0 4px;
    color: #14532d;
    font-size: 24px;
    font-weight: 800;
}
.panel-subtitle {
    margin: 0 0 18px;
    color: #64748b;
    font-size: 14px;
}
button.primary {
    background: #16a34a !important;
    border-color: #16a34a !important;
}
button.primary:hover {
    background: #15803d !important;
}
textarea, input {
    border-radius: 14px !important;
}


.developer-card {
    display: flex;
    align-items: center;
    gap: 18px;
    max-width: 760px;
    margin: 28px auto 8px;
    padding: 20px 22px;
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 70%);
    box-shadow: 0 10px 28px rgba(15,23,42,.06);
    transition: transform .25s ease, box-shadow .25s ease;
    animation: developerIn .55s ease both;
}
.developer-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 34px rgba(22,163,74,.12);
}
.developer-avatar {
    width: 64px;
    height: 64px;
    flex: 0 0 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background: #16a34a;
    font-size: 30px;
    box-shadow: 0 8px 20px rgba(22,163,74,.18);
}
.developer-content {
    min-width: 0;
    text-align: left;
}
.developer-label {
    color: #16a34a;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 3px;
}
.developer-name {
    color: #14532d;
    font-size: 17px;
    font-weight: 800;
}
.developer-email {
    margin-top: 4px;
    font-size: 13px;
}
.developer-email a {
    color: #166534;
    text-decoration: none;
}
.developer-email a:hover {
    text-decoration: underline;
}
.developer-bio {
    margin-top: 7px;
    color: #64748b;
    font-size: 13px;
    line-height: 1.5;
}
@keyframes developerIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 600px) {
    .developer-card {
        align-items: flex-start;
        padding: 16px;
    }
    .developer-avatar {
        width: 52px;
        height: 52px;
        flex-basis: 52px;
        font-size: 24px;
    }
    .developer-name {
        font-size: 14px;
    }
}

@keyframes heroIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes panelIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes floatMark {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
}
@media (max-width: 760px) {
    .hero h1 { font-size: 28px; }
    .nav-btn { min-height: 46px !important; font-size: 13px !important; }
}
"""


THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "sans-serif"],
)


with gr.Blocks(title=APP_TITLE) as demo:

    language_selector = gr.Radio(
        choices=[
            ("🇺🇿 O'zbek", "uz"),
            ("🇬🇧 English", "en"),
            ("🇰🇷 한국어", "ko"),
        ],
        value="uz",
        show_label=False,
    )

    hero = gr.HTML(hero_html("uz"))

    # -----------------------------------------------------------------
    # Animated multilingual navigation
    # The navigation labels change together with the selected UI language.
    # -----------------------------------------------------------------
    with gr.Row(elem_classes="voice-nav"):
        nav_tts = gr.Button(I18N["uz"]["nav_tts"], elem_classes="nav-btn active")
        nav_stt = gr.Button(I18N["uz"]["nav_stt"], elem_classes="nav-btn")
        nav_clone = gr.Button(I18N["uz"]["nav_clone"], elem_classes="nav-btn")
        nav_fix = gr.Button(I18N["uz"]["nav_fix"], elem_classes="nav-btn")

    def nav_update(language, active):
        t = I18N[language]
        labels = [t["nav_tts"], t["nav_stt"], t["nav_clone"], t["nav_fix"]]
        outputs = []

        for i, label in enumerate(labels):
            cls = "nav-btn active" if i == active else "nav-btn"
            outputs.append(gr.Button(value=label, elem_classes=cls))

        visibility = [
            gr.Group(visible=active == 0),
            gr.Group(visible=active == 1),
            gr.Group(visible=active == 2),
            gr.Group(visible=active == 3),
        ]
        return outputs + visibility

    def nav_click(active):
        # Return only visibility updates. Labels are updated by language_selector.
        return [
            gr.Group(visible=active == 0),
            gr.Group(visible=active == 1),
            gr.Group(visible=active == 2),
            gr.Group(visible=active == 3),
        ]

    # -------------------------------------------------------------
    # Text → Speech
    # -------------------------------------------------------------
    with gr.Group(visible=True, elem_classes="panel-card") as tts_panel:
        tts_heading = gr.HTML(
            f'<div class="panel-title">{I18N["uz"]["nav_tts"]}</div>'
            f'<div class="panel-subtitle">{I18N["uz"]["panel_tts"]}</div>'
        )

        tts_language = gr.Dropdown(
            choices=[
                ("🇺🇿 O'zbek", "uz"),
                ("🇬🇧 English", "en"),
                ("🇰🇷 한국어", "ko"),
            ],
            value="en",
            label=I18N["uz"]["language"],
        )

        tts_text = gr.Textbox(
            label=I18N["uz"]["text"],
            placeholder="Enter text / Matn kiriting / 텍스트를 입력하세요...",
            lines=8,
        )

        tts_info = gr.Markdown(I18N["uz"]["tts_info"])

        tts_button = gr.Button(
            "Ovoz yaratish",
            variant="primary",
            size="lg",
        )

        tts_output = gr.Audio(
            label=I18N["uz"]["output"],
            type="filepath",
        )

        tts_button.click(
            text_to_speech,
            inputs=[tts_text, tts_language],
            outputs=tts_output,
        )

    # -------------------------------------------------------------
    # Speech → Text
    # -------------------------------------------------------------
    with gr.Group(visible=False, elem_classes="panel-card") as stt_panel:
        stt_heading = gr.HTML(
            f'<div class="panel-title">{I18N["uz"]["nav_stt"]}</div>'
            f'<div class="panel-subtitle">{I18N["uz"]["panel_stt"]}</div>'
        )

        stt_language = gr.Dropdown(
            choices=[
                ("🇺🇿 O'zbek", "uz"),
                ("🇬🇧 English", "en"),
                ("🇰🇷 한국어", "ko"),
            ],
            value="en",
            label=I18N["uz"]["language"],
        )

        stt_audio = gr.Audio(
            label=I18N["uz"]["audio"],
            type="filepath",
            sources=["upload", "microphone"],
        )

        stt_button = gr.Button(
            "Transcribe",
            variant="primary",
            size="lg",
        )

        stt_output = gr.Textbox(
            label=I18N["uz"]["transcript"],
            lines=12,
            interactive=False,
        )

        stt_button.click(
            speech_to_text,
            inputs=[stt_audio, stt_language],
            outputs=stt_output,
        )

    # -------------------------------------------------------------
    # Voice Cloning
    # -------------------------------------------------------------
    with gr.Group(visible=False, elem_classes="panel-card") as clone_panel:
        clone_heading = gr.HTML(
            f'<div class="panel-title">{I18N["uz"]["nav_clone"]}</div>'
            f'<div class="panel-subtitle">{I18N["uz"]["panel_clone"]}</div>'
        )

        clone_info = gr.Markdown(I18N["uz"]["clone_info"])

        clone_language = gr.Dropdown(
            choices=[
                ("🇬🇧 English", "en"),
                ("🇰🇷 한국어", "ko"),
                ("🇺🇿 O'zbek", "uz"),
            ],
            value="en",
            label=I18N["uz"]["language"],
        )

        clone_reference = gr.Audio(
            label=I18N["uz"]["reference"],
            type="filepath",
            sources=["upload", "microphone"],
        )

        clone_text = gr.Textbox(
            label=I18N["uz"]["text"],
            placeholder="Enter the text for voice cloning...",
            lines=7,
        )

        clone_disabled = gr.Button(
            "Voice cloning model — coming next",
            interactive=False,
        )

        clone_responsible = gr.Markdown(I18N["uz"]["responsible"])

    # -------------------------------------------------------------
    # Audio Enhancement
    # -------------------------------------------------------------
    with gr.Group(visible=False, elem_classes="panel-card") as fix_panel:
        fix_heading = gr.HTML(
            f'<div class="panel-title">{I18N["uz"]["nav_fix"]}</div>'
            f'<div class="panel-subtitle">{I18N["uz"]["panel_fix"]}</div>'
        )

        fix_audio = gr.Audio(
            label=I18N["uz"]["audio"],
            type="filepath",
            sources=["upload", "microphone"],
        )

        with gr.Row():
            fix_denoise = gr.Checkbox(
                value=True,
                label=I18N["uz"]["denoise"],
            )
            fix_trim = gr.Checkbox(
                value=True,
                label=I18N["uz"]["trim"],
            )

        fix_button = gr.Button(
            "Audio yaxshilash",
            variant="primary",
            size="lg",
        )

        fix_output = gr.Audio(
            label=I18N["uz"]["output"],
            type="filepath",
        )

        fix_button.click(
            audio_fix,
            inputs=[fix_audio, fix_denoise, fix_trim],
            outputs=fix_output,
        )


    # -----------------------------------------------------------------
    # Developer / Portfolio
    # -----------------------------------------------------------------
    developer = gr.HTML(
        f"""
        <div class="developer-card">
            <div class="developer-avatar">
                {f'<img src="{PORTFOLIO_DATA}" alt="Developer">' if PORTFOLIO_DATA else '<span>👤</span>'}
            </div>
            <div class="developer-content">
                <div class="developer-label">{I18N["uz"]["developer"]}</div>
                <div class="developer-name">TOJIBOEV IKROMJON MAHKHAMBOY UGLI</div>
                <div class="developer-email">
                    📧 <a href="mailto:ikromjonkorealife@gmail.com">ikromjonkorealife@gmail.com</a>
                </div>
                <div class="developer-bio">{I18N["uz"]["developer_bio"]}</div>
            </div>
        </div>
        """
    )

    footer = gr.Markdown(
        f"""
        <div style="text-align:center;margin-top:24px;padding:12px;color:#64748b;">
            {I18N["uz"]["footer"]}
        </div>
        """
    )

    # -----------------------------------------------------------------
    # UI language + navigation synchronization
    # -----------------------------------------------------------------
    def language_update(language):
        t = I18N[language]

        return [
            hero_html(language),
            gr.Button(value=t["nav_tts"], elem_classes="nav-btn active"),
            gr.Button(value=t["nav_stt"], elem_classes="nav-btn"),
            gr.Button(value=t["nav_clone"], elem_classes="nav-btn"),
            gr.Button(value=t["nav_fix"], elem_classes="nav-btn"),

            gr.HTML(
                f'<div class="panel-title">{t["nav_tts"]}</div>'
                f'<div class="panel-subtitle">{t["panel_tts"]}</div>'
            ),
            gr.Dropdown(label=t["language"], value=tts_language.value),
            gr.Textbox(label=t["text"], placeholder=t["tts_placeholder"]),
            t["tts_info"],
            gr.Button(value=t["generate"], variant="primary", size="lg"),
            gr.Audio(label=t["output"]),

            gr.HTML(
                f'<div class="panel-title">{t["nav_stt"]}</div>'
                f'<div class="panel-subtitle">{t["panel_stt"]}</div>'
            ),
            gr.Dropdown(label=t["language"], value=stt_language.value),
            gr.Audio(label=t["audio"]),
            gr.Button(value=t["transcribe"], variant="primary", size="lg"),
            gr.Textbox(label=t["transcript"]),

            gr.HTML(
                f'<div class="panel-title">{t["nav_clone"]}</div>'
                f'<div class="panel-subtitle">{t["panel_clone"]}</div>'
            ),
            t["clone_info"],
            gr.Dropdown(label=t["language"], value=clone_language.value),
            gr.Audio(label=t["reference"]),
            gr.Textbox(label=t["text"], placeholder=t["clone_placeholder"]),
            gr.Button(value=t["coming_next"], interactive=False),
            t["responsible"],

            gr.HTML(
                f'<div class="panel-title">{t["nav_fix"]}</div>'
                f'<div class="panel-subtitle">{t["panel_fix"]}</div>'
            ),
            gr.Audio(label=t["audio"]),
            gr.Checkbox(label=t["denoise"]),
            gr.Checkbox(label=t["trim"]),
            gr.Button(value=t["enhance"], variant="primary", size="lg"),
            gr.Audio(label=t["output"]),

            gr.HTML(
                f"""
                <div class="developer-card">
                    <div class="developer-avatar">
                        {f'<img src="{PORTFOLIO_DATA}" alt="Developer">' if PORTFOLIO_DATA else '<span>👤</span>'}
                    </div>
                    <div class="developer-content">
                        <div class="developer-label">{t["developer"]}</div>
                        <div class="developer-name">TOJIBOEV IKROMJON MAHKHAMBOY UGLI</div>
                        <div class="developer-email">
                            📧 <a href="mailto:ikromjonkorealife@gmail.com">ikromjonkorealife@gmail.com</a>
                        </div>
                        <div class="developer-bio">{t["developer_bio"]}</div>
                    </div>
                </div>
                """
            ),
            gr.Markdown(
                f"""
                <div style="text-align:center;margin-top:24px;padding:12px;color:#64748b;">
                    {t["footer"]}
                </div>
                """
            ),
        ]

    language_selector.change(
        language_update,
        inputs=language_selector,
        outputs=[
            hero,
            nav_tts, nav_stt, nav_clone, nav_fix,
            tts_heading, tts_language, tts_text, tts_info, tts_button, tts_output,
            stt_heading, stt_language, stt_audio, stt_button, stt_output,
            clone_heading, clone_info, clone_language, clone_reference, clone_text, clone_disabled, clone_responsible,
            fix_heading, fix_audio, fix_denoise, fix_trim, fix_button, fix_output,
            developer, footer,
        ],
    )

    # Navigation clicks switch the visible panel and animate the active item.
    # Each click also uses the current UI language for the labels.
    def select_tts(lang): return nav_update(lang, 0)
    def select_stt(lang): return nav_update(lang, 1)
    def select_clone(lang): return nav_update(lang, 2)
    def select_fix(lang): return nav_update(lang, 3)

    nav_outputs = [
        nav_tts, nav_stt, nav_clone, nav_fix,
        tts_panel, stt_panel, clone_panel, fix_panel
    ]

    nav_tts.click(select_tts, inputs=language_selector, outputs=nav_outputs)
    nav_stt.click(select_stt, inputs=language_selector, outputs=nav_outputs)
    nav_clone.click(select_clone, inputs=language_selector, outputs=nav_outputs)
    nav_fix.click(select_fix, inputs=language_selector, outputs=nav_outputs)
if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=THEME, css=CSS)
