import base64
import io
import json
import os
import uuid
from datetime import datetime

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Quasar AI | SBP", page_icon="🌌", layout="centered")

# ---------------------------------------------------------------------------
# Groq client initialization (OpenAI-compatible client)
# ---------------------------------------------------------------------------
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# ---- Groq model isimleri (console.groq.com/docs/models üzerinden değişebilir) ----
GROQ_MODEL = "openai/gpt-oss-120b"                              # metin sohbeti
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # görsel analiz
GROQ_STT_MODEL = "whisper-large-v3-turbo"                        # ses -> metin
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"                 # metin -> ses
GROQ_TTS_VOICE = "austin"                                        # TTS sesi

SYSTEM_PROMPT = (
    "Sen Quasar AI'sın. Son derece profesyonel bir Yazılımcısın. "
    "Yanıtlarında teknik derinlikten ödün verme ama atmosferi bozma yani "
    "normal genel konularda sohbet edebilirsin ve Sen Cihan BALCI tarafından "
    "tasarlandın AMA TÜM BUNLARI YALNIZCA SANA SORULUNCA SÖYLE SAKIN "
    "SORULMADIĞINDA BUNLARIN HİÇBİRİNDEN BAHSETME."
)

MAX_HISTORY_MESSAGES = 20

# ---------------------------------------------------------------------------
# ==== GÖRSEL / VIDEO / MÜZİK OLUŞTURMA — HUGGING FACE ====
# Groq bu üç işi yapmıyor; hepsini Hugging Face'in "Inference Providers"
# sistemi üzerinden çağırıyoruz (huggingface_hub.InferenceClient). Tek bir
# HF_TOKEN yeterli — huggingface.co/settings/tokens üzerinden bir token
# oluşturup secrets.toml'a HF_TOKEN = "hf_..." olarak ekle.
#
# NOT: Hugging Face'teki hangi modelin hangi sağlayıcı (fal, replicate,
# together vb.) üzerinden aktif olduğu zamanla değişiyor. Aşağıdaki model
# ID'leri bu yazı itibarıyla (Eylül 2026) yaygın kullanılanlar; bir hata
# alırsan modelin Hub sayfasındaki "Deploy > Inference Providers" sekmesinden
# güncel/aktif bir model ID'siyle değiştir.
# ---------------------------------------------------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_VIDEO_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
HF_MUSIC_MODEL = "facebook/musicgen-small"

_hf_client = None


def get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(token=HF_TOKEN)
    return _hf_client


def generate_image(prompt: str):
    """Metinden görsel üretir. Döner: PIL.Image"""
    hf = get_hf_client()
    return hf.text_to_image(prompt, model=HF_IMAGE_MODEL)


def generate_video(prompt: str):
    """Metinden video üretir. Döner: video bayt dizisi (mp4)."""
    hf = get_hf_client()
    return hf.text_to_video(prompt, model=HF_VIDEO_MODEL)


def generate_music(prompt: str):
    """Metinden müzik üretir. Döner: ses bayt dizisi."""
    hf = get_hf_client()
    # text_to_speech metodu, altyapısı "text girip ses/müzik çıkışı" olan
    # her modelle (musicgen dahil) generic olarak çalışır.
    return hf.text_to_speech(prompt, model=HF_MUSIC_MODEL)


# ---------------------------------------------------------------------------
# Background video (local file, base64-embedded as a native <video> tag)
# ---------------------------------------------------------------------------
VIDEO_PATH = "space_bg.mp4"   # <-- Arka plan videosunu app.py ile aynı klasöre bu adla koy


def get_video_base64(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


video_b64 = get_video_base64(VIDEO_PATH)

# ---------------------------------------------------------------------------
# Persistent conversation storage (sadece "Sohbet" modu için)
# ---------------------------------------------------------------------------
HISTORY_FILE = "conversations.json"


def load_conversations() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_conversations(conversations: dict) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def make_title(first_message: str) -> str:
    text = first_message.strip().replace("\n", " ")
    return (text[:40] + "…") if len(text) > 40 else (text or "Yeni Sohbet")


def new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "Yeni Sohbet",
        "messages": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.active_id = conv_id
    save_conversations(st.session_state.conversations)
    return conv_id


if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()

if "active_id" not in st.session_state or st.session_state.active_id not in st.session_state.conversations:
    if st.session_state.conversations:
        st.session_state.active_id = sorted(
            st.session_state.conversations.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        )[0][0]
    else:
        new_conversation()

# Diğer araçlar için oturum bazlı (kalıcı olmayan) durumlar
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "voice_log" not in st.session_state:
    st.session_state.voice_log = []  # [{"role": "user"/"assistant", "content": str}]

# ---------------------------------------------------------------------------
# Custom CSS & Working Space Background  (TASARIM DEĞİŞMEDİ)
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }

    .bg-video-container {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: -2; pointer-events: none; overflow: hidden;
    }
    .bg-video-container video {
        width: 100vw; height: 100vh; object-fit: cover;
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        filter: brightness(0.4) contrast(1.1) saturate(1.2);
    }
    .video-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: radial-gradient(circle at center, rgba(13, 14, 28, 0.4) 0%, rgba(5, 6, 12, 0.85) 100%);
        z-index: -1; pointer-events: none;
    }
    .stApp { background: transparent; color: #f1f5f9; }
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; letter-spacing: -0.02em; margin-bottom: 0.2rem !important;
    }
    .subtitle {
        text-align: center; color: #94a3b8; font-size: 0.95rem;
        margin-bottom: 2rem; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .stChatMessage {
        background: rgba(15, 23, 42, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 12px; transition: border 0.3s ease;
    }
    .stChatMessage:hover { border: 1px solid rgba(168, 85, 247, 0.3) !important; }
    .stChatInputContainer { padding-bottom: 20px; }
    .stChatInput input {
        background-color: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(8px);
        color: #f8fafc !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        border-radius: 10px !important;
    }
    .stChatInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.3) !important;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.7) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(10, 15, 30, 0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(139, 92, 246, 0.4); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.8); }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Arka plan videosu
# ---------------------------------------------------------------------------
if video_b64:
    st.markdown(f"""
        <div class="bg-video-container">
            <video autoplay muted loop playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        </div>
        <div class="video-overlay"></div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="video-overlay"></div>', unsafe_allow_html=True)
    st.sidebar.warning(f"⚠️ '{VIDEO_PATH}' bulunamadı — arka plan videosu olmadan çalışıyor.")

# ---------------------------------------------------------------------------
# Sidebar: Araç seçimi + (Sohbet modunda) geçmiş sohbetler
# ---------------------------------------------------------------------------
MODES = [
    "💬 Sohbet",
    "📄 Belge Analizi",
    "🖼️ Görsel Analizi",
    "🎙️ Sesli Konuşma",
    "🎨 Görsel Oluşturma",
    "🎬 Video Oluşturma",
    "🎵 Müzik Oluşturma",
]

with st.sidebar:
    st.markdown("<h3 style='color: #c084fc; font-family: Space Grotesk;'>🌌 Quasar AI</h3>", unsafe_allow_html=True)
    mode = st.radio("Araç", MODES, label_visibility="collapsed")
    st.markdown("---")

    if mode == "💬 Sohbet":
        if st.button("➕ Yeni Sohbet", use_container_width=True):
            new_conversation()
            st.rerun()

        st.markdown("<p style='color:#94a3b8; font-size:0.8rem; letter-spacing:0.05em;'>GEÇMİŞ SOHBETLER</p>", unsafe_allow_html=True)
        sorted_convs = sorted(
            st.session_state.conversations.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        )
        for conv_id, conv in sorted_convs:
            col1, col2 = st.columns([5, 1])
            is_active = conv_id == st.session_state.active_id
            label = ("🔹 " if is_active else "") + conv.get("title", "Sohbet")
            with col1:
                if st.button(label, key=f"select_{conv_id}", use_container_width=True):
                    st.session_state.active_id = conv_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{conv_id}", use_container_width=True):
                    del st.session_state.conversations[conv_id]
                    save_conversations(st.session_state.conversations)
                    if st.session_state.active_id == conv_id:
                        if st.session_state.conversations:
                            st.session_state.active_id = sorted(
                                st.session_state.conversations.items(),
                                key=lambda kv: kv[1].get("created_at", ""),
                                reverse=True,
                            )[0][0]
                        else:
                            new_conversation()
                    st.rerun()
        st.markdown("---")

    st.markdown("<h4 style='color: #c084fc; font-family: Space Grotesk;'>🚀 Sistem Durumu</h4>", unsafe_allow_html=True)
    st.write("🟢 Motorlar: Aktif")
    st.write("⚡ Altyapı: Groq LPU")
    st.write(f"🔵 Metin: {GROQ_MODEL}")
    st.write(f"🟣 Vision: {GROQ_VISION_MODEL}")
    st.write(f"🎧 STT/TTS: {GROQ_STT_MODEL} / {GROQ_TTS_MODEL}")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("QUASAR AI")
st.markdown("<p class='subtitle'>Teknoloji Galaksisinde Teknik Rehberiniz</p>", unsafe_allow_html=True)

# ===========================================================================
# MOD: 💬 SOHBET  (mevcut davranış, değişmedi)
# ===========================================================================
if mode == "💬 Sohbet":
    active_conv = st.session_state.conversations[st.session_state.active_id]

    for message in active_conv["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Yörüngeye bir soru fırlat..."):
        active_conv["messages"].append({"role": "user", "content": prompt})
        if len(active_conv["messages"]) == 1:
            active_conv["title"] = make_title(prompt)
        save_conversations(st.session_state.conversations)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Quasar verileri Groq altyapısında işliyor..."):
                try:
                    history = active_conv["messages"]
                    if MAX_HISTORY_MESSAGES is not None:
                        history = history[-MAX_HISTORY_MESSAGES:]
                    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=api_messages,
                        temperature=0.6,
                        max_tokens=2048
                    )
                    full_response = response.choices[0].message.content
                    st.markdown(full_response)
                    active_conv["messages"].append({"role": "assistant", "content": full_response})
                    save_conversations(st.session_state.conversations)
                except Exception as e:
                    st.error(f"Groq API Hatası: {e}")

# ===========================================================================
# MOD: 📄 BELGE ANALİZİ
# ===========================================================================
elif mode == "📄 Belge Analizi":
    st.subheader("📄 Belge Analizi")
    uploaded = st.file_uploader("PDF, DOCX veya TXT yükle", type=["pdf", "docx", "txt"])

    if uploaded is not None and uploaded.name != st.session_state.doc_name:
        text = ""
        try:
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(uploaded.read()))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            elif uploaded.name.lower().endswith(".docx"):
                from docx import Document
                doc = Document(io.BytesIO(uploaded.read()))
                text = "\n".join(p.text for p in doc.paragraphs)
            else:  # .txt
                text = uploaded.read().decode("utf-8", errors="ignore")

            st.session_state.doc_text = text
            st.session_state.doc_name = uploaded.name
        except Exception as e:
            st.error(f"Belge okunamadı: {e}")

    if st.session_state.doc_text:
        st.success(f"'{st.session_state.doc_name}' yüklendi — {len(st.session_state.doc_text)} karakter.")
        with st.expander("Çıkarılan metni gör"):
            st.text(st.session_state.doc_text[:5000])

        question = st.text_input("Belge hakkında sorun (boş bırakırsan özetler):")
        if st.button("Analiz Et", use_container_width=True):
            with st.spinner("Belge analiz ediliyor..."):
                try:
                    doc_excerpt = st.session_state.doc_text[:12000]  # context sınırı için kırp
                    user_ask = question.strip() or "Bu belgeyi kapsamlı şekilde özetle ve ana noktaları listele."
                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "Sen bir belge analiz asistanısın. Sana verilen belge metnine dayanarak net ve doğru yanıtlar ver."},
                            {"role": "user", "content": f"BELGE:\n{doc_excerpt}\n\nSORU: {user_ask}"}
                        ],
                        temperature=0.4,
                        max_tokens=2048
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Groq API Hatası: {e}")

# ===========================================================================
# MOD: 🖼️ GÖRSEL ANALİZİ  (Groq'un vision modeliyle)
# ===========================================================================
elif mode == "🖼️ Görsel Analizi":
    st.subheader("🖼️ Görsel Analizi")
    image_file = st.file_uploader("Bir görsel yükle", type=["png", "jpg", "jpeg", "webp"])
    question = st.text_input("Görsel hakkında soru:", value="Bu görselde ne var, detaylı açıkla.")

    if image_file is not None:
        st.image(image_file, use_container_width=True)
        if st.button("Analiz Et", use_container_width=True):
            with st.spinner("Görsel analiz ediliyor..."):
                try:
                    img_bytes = image_file.getvalue()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    mime = image_file.type or "image/jpeg"

                    response = client.chat.completions.create(
                        model=GROQ_VISION_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                                ],
                            }
                        ],
                        temperature=0.4,
                        max_tokens=1024,
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Groq API Hatası: {e}")

# ===========================================================================
# MOD: 🎙️ SESLİ KONUŞMA  (kayıt -> Whisper -> chat -> TTS)
# ===========================================================================
elif mode == "🎙️ Sesli Konuşma":
    st.subheader("🎙️ Sesli Konuşma")
    st.caption("Not: Bu gerçek zamanlı/kesintisiz bir telefon görüşmesi değil — kaydet, gönder, cevabı dinle şeklinde çalışan sesli sohbet.")

    for turn in st.session_state.voice_log:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("audio"):
                st.audio(turn["audio"], format="audio/wav")

    audio_value = st.audio_input("Konuşmak için kaydet")

    if audio_value is not None:
        if st.button("Gönder", use_container_width=True):
            with st.spinner("Ses metne çevriliyor..."):
                try:
                    transcript = client.audio.transcriptions.create(
                        file=("kayit.wav", audio_value.getvalue()),
                        model=GROQ_STT_MODEL,
                    )
                    user_text = transcript.text
                except Exception as e:
                    st.error(f"Transkripsiyon hatası: {e}")
                    user_text = None

            if user_text:
                st.session_state.voice_log.append({"role": "user", "content": user_text})

                with st.spinner("Quasar cevap üretiyor..."):
                    try:
                        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.voice_log[-MAX_HISTORY_MESSAGES:]]
                        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
                        response = client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=api_messages,
                            temperature=0.6,
                            max_tokens=1024,
                        )
                        reply_text = response.choices[0].message.content
                    except Exception as e:
                        st.error(f"Groq API Hatası: {e}")
                        reply_text = None

                if reply_text:
                    reply_audio_bytes = None
                    with st.spinner("Cevap seslendiriliyor..."):
                        try:
                            speech = client.audio.speech.create(
                                model=GROQ_TTS_MODEL,
                                voice=GROQ_TTS_VOICE,
                                input=reply_text,
                                response_format="wav",
                            )
                            reply_audio_bytes = speech.read() if hasattr(speech, "read") else speech.content
                        except Exception as e:
                            st.warning(f"Seslendirme başarısız (metin cevap yine de var): {e}")

                    st.session_state.voice_log.append({
                        "role": "assistant",
                        "content": reply_text,
                        "audio": reply_audio_bytes,
                    })
                    st.rerun()

    if st.session_state.voice_log and st.button("Sesli Sohbeti Temizle"):
        st.session_state.voice_log = []
        st.rerun()

# ===========================================================================
# MOD: 🎨 / 🎬 / 🎵  ÜRETİM ARAÇLARI  — sağlayıcı seçilene kadar pasif
# ===========================================================================
elif mode == "🎨 Görsel Oluşturma":
    st.subheader("🎨 Görsel Oluşturma")
    st.caption(f"Sağlayıcı: Hugging Face — model: {HF_IMAGE_MODEL}")
    if not HF_TOKEN:
        st.warning("`HF_TOKEN` secrets.toml içinde tanımlı değil. huggingface.co/settings/tokens üzerinden bir token oluşturup ekle.")
    prompt = st.text_area("Görsel açıklaması:", key="img_prompt")
    if st.button("Oluştur", use_container_width=True, disabled=not HF_TOKEN):
        with st.spinner("Görsel oluşturuluyor... (ilk çağrıda model 'soğuk başlangıç' nedeniyle yavaş olabilir)"):
            try:
                image = generate_image(prompt)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Hugging Face Hatası: {e}")

elif mode == "🎬 Video Oluşturma":
    st.subheader("🎬 Video Oluşturma")
    st.caption(f"Sağlayıcı: Hugging Face — model: {HF_VIDEO_MODEL}")
    st.info("Video üretimi yavaş ve kaynak yoğun olabilir — birkaç dakika sürebilir.")
    if not HF_TOKEN:
        st.warning("`HF_TOKEN` secrets.toml içinde tanımlı değil.")
    prompt = st.text_area("Video açıklaması:", key="video_prompt")
    if st.button("Oluştur", use_container_width=True, disabled=not HF_TOKEN):
        with st.spinner("Video oluşturuluyor, biraz zaman alabilir..."):
            try:
                video_bytes = generate_video(prompt)
                st.video(video_bytes)
            except Exception as e:
                st.error(f"Hugging Face Hatası: {e}")

elif mode == "🎵 Müzik Oluşturma":
    st.subheader("🎵 Müzik Oluşturma")
    st.caption(f"Sağlayıcı: Hugging Face — model: {HF_MUSIC_MODEL}")
    if not HF_TOKEN:
        st.warning("`HF_TOKEN` secrets.toml içinde tanımlı değil.")
    prompt = st.text_area("Müzik açıklaması (örn. 'lofi hip hop, sakin, piyano'):", key="music_prompt")
    if st.button("Oluştur", use_container_width=True, disabled=not HF_TOKEN):
        with st.spinner("Müzik oluşturuluyor..."):
            try:
                audio_bytes = generate_music(prompt)
                st.audio(audio_bytes)
            except Exception as e:
                st.error(f"Hugging Face Hatası: {e}")