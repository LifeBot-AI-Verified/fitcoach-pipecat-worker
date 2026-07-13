import asyncio
import json
import fractions
import math
import os
import re
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
import av
from aiortc import AudioStreamTrack

from aiohttp import web
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

load_dotenv(dotenv_path=Path(".env"))

logger.remove()
logger.add(lambda message: print(message, end=""), level="WARNING")

openai_client = OpenAI()

ACTIVE_CALL_CONNECTIONS: dict[str, SmallWebRTCConnection] = {}
ACTIVE_CALL_AUDIO_TASKS: dict[str, asyncio.Task] = {}

DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_TTS_VOICE = "coral"
DEFAULT_OPENAI_TTS_INSTRUCTIONS = (
    "Habla en español natural y neutro, con tono claro y cercano. "
    "Mantén ritmo normal de conversación telefónica. "
    "No sobreactúes, no alargues las palabras y no fuerces acentos ni emociones."
)
FITCOACH_VOICE_REPLY_SYSTEM_PROMPT = (
    "Eres FitCoach AI, una entrenadora personal y nutricional por voz. "
    "Devuelve solo un objeto JSON valido con las claves spoken_reply, written_recap "
    "y opcionalmente context_updates. "
    "spoken_reply es para una llamada telefonica: debe ser breve, natural, claro y conversacional, "
    "sin markdown y sin listas largas. Si das una rutina, spoken_reply debe ser una version hablada compacta "
    "y util para seguir en llamada, no solo decir que lo dejas en el chat. "
    "written_recap es para WhatsApp: debe ser mas completo, util y estructurado en texto claro. "
    "context_updates, si aparece, debe ser un objeto con solo estas claves cuando haya datos nuevos: "
    "goal, muscle_group, duration, level, equipment, place, limitations, preferences. "
    "Si el usuario pide una rutina, spoken_reply debe cubrir los bloques principales en unos 45 a 90 segundos; "
    "written_recap debe incluir ejercicios, series, repeticiones o tiempos, descansos, intensidad aproximada "
    "y recomendaciones de seguridad cuando aplique. "
    "Usa el historial y el contexto fitness actual de la llamada para entender ajustes como hacerlo mas corto, "
    "cambiar un ejercicio o adaptar material. "
    "Si faltan datos importantes para una rutina, haz como maximo una pregunta breve antes de proponer; "
    "si la peticion ya trae suficiente informacion, responde directamente. "
    "Si el usuario pide parar, responde brevemente que paras o que esperas nueva instruccion, sin crear una rutina nueva. "
    "Si menciona dolor, lesion, embarazo, mareo o una condicion medica, baja la intensidad en ambas respuestas "
    "y recomienda consultar a un profesional. "
    "No incluyas texto fuera del JSON."
)
FALLBACK_SPOKEN_REPLY = (
    "Perdona, he tenido un problema preparando la respuesta completa. "
    "Te dejo una recomendacion breve por el chat para que puedas seguir."
)
FALLBACK_WRITTEN_RECAP = (
    "No he podido preparar el detalle completo esta vez. "
    "Como recomendacion general, entrena con intensidad moderada, evita dolor o mareo, "
    "y consulta con un profesional si tienes una lesion, condicion medica o dudas de seguridad."
)
HELLO_OPENAI_WAV_PATH = Path("hello_openai.wav")
HELLO_OPENAI_META_PATH = Path("hello_openai.meta.json")
HELLO_OPENAI_TEXT = "Hola, soy FitCoach. ¿En qué puedo ayudarte?"
HELLO_OPENAI_PREROLL_SECONDS = 0.4
REPLY_PREROLL_SECONDS = 0.8
SUPPORTED_TTS_WAV_SAMPLE_RATES = {24000, 48000}
MAX_CONVERSATION_HISTORY_TURNS = 6
FITNESS_CONTEXT_FIELDS = {
    "goal",
    "muscle_group",
    "duration",
    "level",
    "equipment",
    "place",
    "limitations",
    "preferences",
}
STOP_REQUEST_PHRASES = (
    "parate",
    "párate",
    "detente",
    "corta",
    "córtalo",
    "no sigas",
    "callate",
    "cállate",
    "silencio",
)
STOP_REQUEST_EXACT_PHRASES = ("para",)


def mask_id(value: str | None) -> str | None:
    if not value:
        return None
    return f"***{value[-4:]}"


def truncate_text(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def get_env_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    return value or default


def filter_sdp_for_whatsapp(sdp: str) -> str:
    lines = sdp.splitlines()
    filtered = []
    for line in lines:
        if line.startswith("a=fingerprint:") and not line.startswith("a=fingerprint:sha-256"):
            continue
        filtered.append(line)
    return "\r\n".join(filtered) + "\r\n"


class BeepAudioTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000, frequency: int = 440):
        super().__init__()
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.samples_per_frame = 960
        self.sample_index = 0

    async def recv(self):
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)

        t = (np.arange(self.samples_per_frame) + self.sample_index) / self.sample_rate
        wave = 0.15 * np.sin(2 * math.pi * self.frequency * t)
        audio = (wave * 32767).astype(np.int16)

        frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.sample_index
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self.sample_index += self.samples_per_frame
        return frame


def get_openai_tts_config() -> dict[str, str]:
    return {
        "model": get_env_value("OPENAI_TTS_MODEL", DEFAULT_OPENAI_TTS_MODEL),
        "voice": get_env_value("OPENAI_TTS_VOICE", DEFAULT_OPENAI_TTS_VOICE),
        "instructions": get_env_value("OPENAI_TTS_INSTRUCTIONS", DEFAULT_OPENAI_TTS_INSTRUCTIONS),
    }


def generate_openai_tts_wav(
    text: str,
    path: str | Path,
    tts_config: dict[str, str] | None = None,
):
    tts_config = tts_config or get_openai_tts_config()

    with openai_client.audio.speech.with_streaming_response.create(
        model=tts_config["model"],
        voice=tts_config["voice"],
        instructions=tts_config["instructions"],
        input=text,
        response_format="wav",
    ) as response:
        response.stream_to_file(str(path))


def add_wav_preroll_silence(path: str | Path, seconds: float = REPLY_PREROLL_SECONDS):
    wav_path = str(path)

    with wave.open(wav_path, "rb") as src:
        channels = src.getnchannels()
        sample_width = src.getsampwidth()
        sample_rate = src.getframerate()
        frames = src.readframes(src.getnframes())

    silence_frames = int(sample_rate * seconds)
    silence = b"\x00" * silence_frames * channels * sample_width

    with wave.open(wav_path, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(sample_rate)
        dst.writeframes(silence + frames)


def is_valid_tts_wav(path: str | Path) -> bool:
    wav_path = Path(path)
    if not wav_path.is_file():
        return False

    try:
        with wave.open(str(wav_path), "rb") as src:
            return (
                src.getnchannels() >= 1
                and src.getsampwidth() == 2
                and src.getframerate() in SUPPORTED_TTS_WAV_SAMPLE_RATES
                and src.getnframes() > 0
            )
    except (EOFError, OSError, wave.Error):
        return False


def get_hello_openai_metadata(tts_config: dict[str, str]) -> dict[str, str]:
    return {
        "text": HELLO_OPENAI_TEXT,
        "model": tts_config["model"],
        "voice": tts_config["voice"],
        "instructions": tts_config["instructions"],
    }


def hello_openai_metadata_matches(expected_metadata: dict[str, str]) -> bool:
    try:
        metadata = json.loads(HELLO_OPENAI_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(metadata, dict):
        return False

    return all(metadata.get(key) == value for key, value in expected_metadata.items())


def write_hello_openai_metadata(metadata: dict[str, str]):
    HELLO_OPENAI_META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_hello_openai_wav() -> str:
    tts_config = get_openai_tts_config()
    expected_metadata = get_hello_openai_metadata(tts_config)

    if is_valid_tts_wav(HELLO_OPENAI_WAV_PATH) and hello_openai_metadata_matches(expected_metadata):
        return str(HELLO_OPENAI_WAV_PATH)

    generate_openai_tts_wav(HELLO_OPENAI_TEXT, HELLO_OPENAI_WAV_PATH, tts_config=tts_config)
    add_wav_preroll_silence(HELLO_OPENAI_WAV_PATH, seconds=HELLO_OPENAI_PREROLL_SECONDS)

    if not is_valid_tts_wav(HELLO_OPENAI_WAV_PATH):
        raise ValueError(f"Generated invalid greeting WAV: {HELLO_OPENAI_WAV_PATH}")

    write_hello_openai_metadata(expected_metadata)

    return str(HELLO_OPENAI_WAV_PATH)


def build_fitcoach_voice_reply_input(
    user_text: str,
    conversation_history: list[dict[str, str]] | None = None,
    session_fitness_context: dict[str, str | list[str]] | None = None,
) -> list[dict[str, str]]:
    input_messages = [
        {
            "role": "system",
            "content": FITCOACH_VOICE_REPLY_SYSTEM_PROMPT,
        }
    ]

    input_messages.append(
        {
            "role": "user",
            "content": (
                "Contexto fitness actual de esta llamada, detectado en turnos previos. "
                "Usalo solo si ayuda y actualizalo con context_updates cuando haya informacion nueva:\n"
                + json.dumps(session_fitness_context or {}, ensure_ascii=False)
            ),
        }
    )

    for turn in (conversation_history or [])[-MAX_CONVERSATION_HISTORY_TURNS:]:
        previous_user_text = turn.get("user_text", "")
        previous_spoken_reply = turn.get("spoken_reply", "")
        previous_written_recap = turn.get("written_recap", "")

        if previous_user_text:
            input_messages.append(
                {
                    "role": "user",
                    "content": previous_user_text,
                }
            )

        if previous_spoken_reply or previous_written_recap:
            input_messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "spoken_reply": previous_spoken_reply,
                            "written_recap": previous_written_recap,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    input_messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    return input_messages


def parse_fitcoach_call_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parse_failed = False

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end > json_start:
            try:
                data = json.loads(text[json_start : json_end + 1])
            except json.JSONDecodeError:
                parse_failed = True
                data = {}
        else:
            parse_failed = True
            data = {}

    if not isinstance(data, dict):
        parse_failed = True
        data = {}

    spoken_reply = data.get("spoken_reply")
    written_recap = data.get("written_recap")
    context_updates = data.get("context_updates")

    if parse_failed or not isinstance(spoken_reply, str) or not spoken_reply.strip():
        spoken_reply = FALLBACK_SPOKEN_REPLY

    if not isinstance(written_recap, str) or not written_recap.strip():
        written_recap = FALLBACK_WRITTEN_RECAP

    if not isinstance(context_updates, dict):
        context_updates = {}

    return {
        "spoken_reply": spoken_reply.strip(),
        "written_recap": written_recap.strip(),
        "context_updates": context_updates,
    }


def normalize_fitness_context_value(value: Any) -> str | list[str] | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None

    if isinstance(value, list):
        normalized_items = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized_item = item.strip()
            if normalized_item:
                normalized_items.append(normalized_item)
        return normalized_items[:12] or None

    return None


def merge_fitness_context_values(
    current_value: str | list[str] | None,
    new_value: str | list[str],
) -> str | list[str]:
    if isinstance(new_value, str):
        return new_value

    existing_items = current_value if isinstance(current_value, list) else []
    merged_items = list(existing_items)

    for item in new_value:
        if item not in merged_items:
            merged_items.append(item)

    return merged_items[:12]


def update_session_fitness_context(
    session_fitness_context: dict[str, str | list[str]],
    context_updates: Any,
) -> None:
    if not isinstance(context_updates, dict):
        return

    for key, value in context_updates.items():
        if not isinstance(key, str):
            continue

        normalized_key = key.strip().lower()
        if normalized_key not in FITNESS_CONTEXT_FIELDS:
            continue

        normalized_value = normalize_fitness_context_value(value)
        if normalized_value is None:
            continue

        session_fitness_context[normalized_key] = merge_fitness_context_values(
            session_fitness_context.get(normalized_key),
            normalized_value,
        )


def add_context_update(updates: dict[str, str | list[str]], key: str, value: str | list[str]) -> None:
    normalized_value = normalize_fitness_context_value(value)
    if normalized_value is None:
        return

    updates[key] = merge_fitness_context_values(updates.get(key), normalized_value)


def infer_fitness_context_updates_from_user_text(user_text: str) -> dict[str, str | list[str]]:
    text = user_text.lower()
    updates: dict[str, str | list[str]] = {}

    duration_match = re.search(r"\b(\d{1,3})\s*(?:minutos?|mins?|min)\b", text)
    if duration_match:
        add_context_update(updates, "duration", f"{duration_match.group(1)} minutos")

    muscle_groups = {
        "espalda": ("espalda",),
        "piernas": ("piernas", "pierna"),
        "pecho": ("pecho",),
        "hombros": ("hombros", "hombro"),
        "brazos": ("brazos", "brazo", "biceps", "bíceps", "triceps", "tríceps"),
        "gluteos": ("gluteos", "glúteos", "gluteo", "glúteo"),
        "abdomen": ("abdomen", "abdominales", "core"),
    }
    for muscle_group, aliases in muscle_groups.items():
        if any(alias in text for alias in aliases):
            add_context_update(updates, "muscle_group", muscle_group)
            break

    if any(phrase in text for phrase in ("en casa", "para casa", "desde casa", "casa")):
        add_context_update(updates, "place", "casa")
    elif "gimnasio" in text or "gym" in text:
        add_context_update(updates, "place", "gimnasio")
    elif any(phrase in text for phrase in ("exterior", "aire libre", "parque")):
        add_context_update(updates, "place", "exterior")

    if re.search(r"\b(?:sin|no tengo|no hay)\s+mancuernas\b", text):
        add_context_update(updates, "equipment", "sin mancuernas")
        add_context_update(updates, "preferences", ["sin mancuernas"])
    elif "mancuernas" in text:
        add_context_update(updates, "equipment", "mancuernas")

    if "sin saltos" in text or "no puedo saltar" in text:
        add_context_update(updates, "preferences", ["sin saltos", "bajo impacto"])
    if "bajo impacto" in text:
        add_context_update(updates, "preferences", ["bajo impacto"])

    if any(phrase in text for phrase in ("mas facil", "más fácil", "facil", "fácil", "principiante")):
        add_context_update(updates, "level", "principiante")
    elif "intermedio" in text:
        add_context_update(updates, "level", "intermedio")
    elif "avanzado" in text:
        add_context_update(updates, "level", "avanzado")

    if any(phrase in text for phrase in ("perder grasa", "bajar grasa", "adelgazar")):
        add_context_update(updates, "goal", "perder grasa")
    elif any(phrase in text for phrase in ("ganar musculo", "ganar músculo", "hipertrofia")):
        add_context_update(updates, "goal", "ganar músculo")
    elif "fuerza" in text:
        add_context_update(updates, "goal", "fuerza")
    elif "movilidad" in text:
        add_context_update(updates, "goal", "movilidad")
    elif "cardio" in text:
        add_context_update(updates, "goal", "cardio")

    if re.search(r"\b(?:dolor|duele|lesion|lesión|embarazo|embarazada|mareo|mareado|mareada)\b", text):
        add_context_update(updates, "limitations", [truncate_text(user_text, limit=180)])

    return updates


def is_stop_request(user_text: str) -> bool:
    normalized_text = re.sub(r"[¡!¿?.,;:]+", " ", user_text.lower())
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    if re.fullmatch(r"(?:por favor )?para(?: (?:por favor|ya))?", normalized_text):
        return True

    if normalized_text in STOP_REQUEST_EXACT_PHRASES:
        return True

    return any(
        re.fullmatch(rf"(?:por favor )?{re.escape(phrase)}(?: por favor)?", normalized_text)
        for phrase in STOP_REQUEST_PHRASES
    )


def build_stop_call_response() -> dict[str, Any]:
    return {
        "spoken_reply": "De acuerdo, paro. Me quedo esperando tu siguiente instruccion.",
        "written_recap": (
            "He parado la propuesta actual. Si quieres, puedes pedirme un ajuste concreto "
            "o una nueva indicacion cuando te venga bien."
        ),
        "context_updates": {},
    }


def generate_fitcoach_call_response(
    user_text: str,
    conversation_history: list[dict[str, str]] | None = None,
    session_fitness_context: dict[str, str | list[str]] | None = None,
) -> dict[str, Any]:
    if is_stop_request(user_text):
        return build_stop_call_response()

    try:
        response = openai_client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=build_fitcoach_voice_reply_input(
                user_text,
                conversation_history,
                session_fitness_context,
            ),
        )
    except Exception as exc:
        print(
            "[pipecat-worker] WhatsApp LLM response generation failed",
            {
                "errorType": type(exc).__name__,
                "error": str(exc)[:500],
            },
            flush=True,
        )
        return {
            "spoken_reply": FALLBACK_SPOKEN_REPLY,
            "written_recap": FALLBACK_WRITTEN_RECAP,
            "context_updates": {},
        }

    return parse_fitcoach_call_response(getattr(response, "output_text", "") or "")


def build_whatsapp_recap_text(written_recap: str) -> str:
    return "Te dejo por escrito el detalle de la llamada:\n\n" + written_recap


def get_wav_duration_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as src:
        sample_rate = src.getframerate()
        if sample_rate <= 0:
            return 0.0
        return src.getnframes() / sample_rate


def write_last_reply_debug(
    call_response: dict[str, Any],
    session_fitness_context: dict[str, str | list[str]] | None = None,
):
    Path("last_reply.txt").write_text(
        json.dumps(
            {
                "spoken_reply": call_response["spoken_reply"],
                "written_recap": call_response["written_recap"],
                "session_fitness_context": session_fitness_context or {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class WavAudioTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, path: str, sample_rate: int = 48000):
        super().__init__()
        self.path = path
        self.sample_rate = sample_rate
        self.samples_per_frame = 960
        self.sample_index = 0
        self.audio = self._load_audio(path)

    def _load_audio(self, path: str):
        with wave.open(path, "rb") as src:
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            sample_rate = src.getframerate()
            frames = src.readframes(src.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

        if sample_width != 2:
            raise ValueError(f"Expected 16-bit WAV, got sample width {sample_width}")

        if sample_rate == self.sample_rate:
            return audio

        if sample_rate == 24000 and self.sample_rate == 48000:
            return np.repeat(audio, 2).astype(np.int16)

        raise ValueError(f"Unsupported WAV sample rate conversion: {sample_rate}Hz to {self.sample_rate}Hz")

    async def recv(self):
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)

        start = self.sample_index
        end = start + self.samples_per_frame
        chunk = self.audio[start:end]

        if len(chunk) < self.samples_per_frame:
            remaining = self.samples_per_frame - len(chunk)
            chunk = np.concatenate([chunk, np.zeros(remaining, dtype=np.int16)])

        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.sample_index
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self.sample_index += self.samples_per_frame

        return frame


class SilenceAudioTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = 960
        self.sample_index = 0

    async def recv(self):
        await asyncio.sleep(self.samples_per_frame / self.sample_rate)

        chunk = np.zeros(self.samples_per_frame, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.sample_index
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self.sample_index += self.samples_per_frame
        return frame


def get_mono_samples_from_frame(frame: av.AudioFrame) -> np.ndarray | None:
    ndarray = frame.to_ndarray()

    if ndarray.size == 0:
        return None

    raw_samples = ndarray.astype(np.int16).reshape(-1)

    if getattr(frame, "layout", None) and str(frame.layout) == "<av.AudioLayout 'stereo'>":
        return raw_samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    return raw_samples


def get_samples_rms_and_peak(samples: np.ndarray) -> tuple[float, float]:
    samples_float = samples.astype(np.float32)
    rms = float(np.sqrt(np.mean(samples_float * samples_float)))
    peak = float(np.max(np.abs(samples_float)))
    return rms, peak


def transcribe_captured_audio(
    *,
    call_id: str,
    captured_chunks: list[np.ndarray],
    wav_path: str,
    sample_rate: int,
) -> str | None:
    if not captured_chunks:
        print(
            "[pipecat-worker] WhatsApp speech capture empty",
            {"callId": call_id},
            flush=True,
        )
        return None

    audio = np.concatenate(captured_chunks).astype(np.int16)

    with wave.open(wav_path, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(sample_rate)
        dst.writeframes(audio.tobytes())

    print(
        "[pipecat-worker] WhatsApp speech wav saved",
        {
            "callId": call_id,
            "path": wav_path,
            "samples": int(audio.size),
            "seconds": round(audio.size / sample_rate, 2),
        },
        flush=True,
    )

    with open(wav_path, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
            file=audio_file,
            language="es",
        )

    text = getattr(transcription, "text", "") or ""

    print(
        "[pipecat-worker] WhatsApp speech transcribed",
        {
            "callId": call_id,
            "transcriptionLength": len(text),
            "transcriptionPreview": text[:160],
        },
        flush=True,
    )

    return text.strip() or None


async def capture_user_utterance(
    *,
    call_id: str,
    audio_track: Any,
    wav_path: str = "last_call_input.wav",
    sample_rate: int = 48000,
    voice_threshold: float = 700.0,
    max_voice_frames: int = 250,
    pre_speech_frames: int = 25,
) -> str | None:
    captured_chunks = []
    rolling_chunks = []
    capturing = False
    voice_frames = 0
    frame_count = 0

    while True:
        frame = await audio_track.recv()
        samples = get_mono_samples_from_frame(frame)
        if samples is None:
            continue

        rms, peak = get_samples_rms_and_peak(samples)

        frame_count += 1

        rolling_chunks.append(samples.copy())
        if len(rolling_chunks) > pre_speech_frames:
            rolling_chunks.pop(0)

        if not capturing and rms > voice_threshold:
            capturing = True
            captured_chunks.extend(rolling_chunks)
            print(
                "[pipecat-worker] WhatsApp speech capture started",
                {
                    "callId": call_id,
                    "frameCount": frame_count,
                    "rms": round(rms, 2),
                    "peak": round(peak, 2),
                },
                flush=True,
            )

        if capturing:
            captured_chunks.append(samples.copy())
            voice_frames += 1

            if voice_frames >= max_voice_frames:
                break

    return transcribe_captured_audio(
        call_id=call_id,
        captured_chunks=captured_chunks,
        wav_path=wav_path,
        sample_rate=sample_rate,
    )


async def wait_for_reply_or_interruption(
    *,
    call_id: str,
    connection: SmallWebRTCConnection,
    audio_track: Any,
    reply_duration_seconds: float,
    wav_path: str = "last_call_input.wav",
    sample_rate: int = 48000,
    voice_threshold: float = 700.0,
    max_voice_frames: int = 250,
    pre_speech_frames: int = 25,
    interruption_voice_frames: int = 3,
) -> str | None:
    print(
        "[pipecat-worker] WhatsApp reply playback monitoring started",
        {
            "callId": call_id,
            "durationSeconds": round(reply_duration_seconds, 2),
        },
        flush=True,
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + reply_duration_seconds + 0.5
    rolling_chunks = []
    loud_frames = 0
    frame_count = 0

    while loop.time() < deadline:
        frame = await audio_track.recv()
        samples = get_mono_samples_from_frame(frame)
        if samples is None:
            continue

        frame_count += 1
        rolling_chunks.append(samples.copy())
        if len(rolling_chunks) > pre_speech_frames:
            rolling_chunks.pop(0)

        rms, peak = get_samples_rms_and_peak(samples)
        if rms > voice_threshold:
            loud_frames += 1
        else:
            loud_frames = 0

        if loud_frames < interruption_voice_frames:
            continue

        print(
            "[pipecat-worker] WhatsApp reply playback interrupted",
            {
                "callId": call_id,
                "frameCount": frame_count,
                "rms": round(rms, 2),
                "peak": round(peak, 2),
            },
            flush=True,
        )

        connection.replace_audio_track(SilenceAudioTrack())
        captured_chunks = list(rolling_chunks)
        silent_frames_after_speech = 0

        while len(captured_chunks) < max_voice_frames:
            frame = await audio_track.recv()
            samples = get_mono_samples_from_frame(frame)
            if samples is None:
                continue
            captured_chunks.append(samples.copy())

            rms, _ = get_samples_rms_and_peak(samples)
            if rms > voice_threshold:
                silent_frames_after_speech = 0
            else:
                silent_frames_after_speech += 1

            if silent_frames_after_speech >= pre_speech_frames:
                break

        interrupted_text = transcribe_captured_audio(
            call_id=call_id,
            captured_chunks=captured_chunks,
            wav_path=wav_path,
            sample_rate=sample_rate,
        )

        print(
            "[pipecat-worker] WhatsApp interruption transcribed",
            {
                "callId": call_id,
                "textPresent": bool(interrupted_text),
                "textPreview": (interrupted_text or "")[:160],
            },
            flush=True,
        )
        return interrupted_text

    print(
        "[pipecat-worker] WhatsApp reply playback completed",
        {
            "callId": call_id,
            "durationSeconds": round(reply_duration_seconds, 2),
        },
        flush=True,
    )
    return None


async def monitor_audio_input(call_id: str, connection: SmallWebRTCConnection, raw_from: str | None = None):
    wav_path = "last_call_input.wav"
    sample_rate = 48000
    voice_threshold = 700.0
    max_voice_frames = 250
    pre_speech_frames = 25

    try:
        await asyncio.sleep(1.0)

        audio_track = connection.audio_input_track()
        if audio_track is None:
            print(
                "[pipecat-worker] WhatsApp audio input unavailable",
                {"callId": call_id},
                flush=True,
            )
            return

        print(
            "[pipecat-worker] WhatsApp audio input STT monitor started",
            {"callId": call_id},
            flush=True,
        )

        conversation_history: list[dict[str, str]] = []
        session_fitness_context: dict[str, str | list[str]] = {}
        turn_index = 0
        pending_user_text: str | None = None

        while True:
            print(
                "[pipecat-worker] WhatsApp turn listening",
                {
                    "callId": call_id,
                    "turnIndex": turn_index + 1,
                    "historyTurns": len(conversation_history),
                    "fitnessContextFields": sorted(session_fitness_context.keys()),
                },
                flush=True,
            )

            if pending_user_text:
                user_text = pending_user_text
                pending_user_text = None
            else:
                user_text = await capture_user_utterance(
                    call_id=call_id,
                    audio_track=audio_track,
                    wav_path=wav_path,
                    sample_rate=sample_rate,
                    voice_threshold=voice_threshold,
                    max_voice_frames=max_voice_frames,
                    pre_speech_frames=pre_speech_frames,
                )

            if not user_text:
                await asyncio.sleep(0.25)
                continue

            turn_index += 1
            update_session_fitness_context(
                session_fitness_context,
                infer_fitness_context_updates_from_user_text(user_text),
            )
            call_response = generate_fitcoach_call_response(
                user_text,
                conversation_history,
                session_fitness_context,
            )
            update_session_fitness_context(session_fitness_context, call_response.get("context_updates"))
            spoken_reply = call_response["spoken_reply"]
            written_recap = call_response["written_recap"]

            print(
                "[pipecat-worker] WhatsApp LLM reply generated",
                {
                    "callId": call_id,
                    "turnIndex": turn_index,
                    "spokenReplyLength": len(spoken_reply),
                    "writtenRecapLength": len(written_recap),
                    "spokenReplyPreview": spoken_reply[:220],
                    "fitnessContextFields": sorted(session_fitness_context.keys()),
                },
                flush=True,
            )

            conversation_history.append(
                {
                    "user_text": user_text,
                    "spoken_reply": spoken_reply,
                    "written_recap": written_recap,
                }
            )
            conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY_TURNS:]

            write_last_reply_debug(call_response, session_fitness_context)

            if raw_from:
                recap_text = build_whatsapp_recap_text(written_recap)
                recap_result = send_whatsapp_text_message(to=raw_from, text=recap_text)

                print(
                    "[pipecat-worker] WhatsApp call recap message sent",
                    {
                        "callId": call_id,
                        "turnIndex": turn_index,
                        "ok": recap_result.get("ok"),
                        "status": recap_result.get("status"),
                        "bodyLength": recap_result.get("bodyLength"),
                        "error": recap_result.get("error"),
                    },
                    flush=True,
                )
            else:
                print(
                    "[pipecat-worker] WhatsApp call recap message skipped",
                    {
                        "callId": call_id,
                        "turnIndex": turn_index,
                        "reason": "missing_raw_from",
                    },
                    flush=True,
                )

            generate_openai_tts_wav(spoken_reply, "reply.wav")
            add_wav_preroll_silence("reply.wav", seconds=REPLY_PREROLL_SECONDS)
            reply_duration_seconds = get_wav_duration_seconds("reply.wav")

            print(
                "[pipecat-worker] WhatsApp reply wav generated",
                {
                    "callId": call_id,
                    "turnIndex": turn_index,
                    "path": "reply.wav",
                    "spokenReplyLength": len(spoken_reply),
                    "durationSeconds": round(reply_duration_seconds, 2),
                    "prerollSilenceSeconds": REPLY_PREROLL_SECONDS,
                },
                flush=True,
            )

            connection_for_reply = ACTIVE_CALL_CONNECTIONS.get(call_id)
            if connection_for_reply:
                connection_for_reply.replace_audio_track(WavAudioTrack("reply.wav"))
                print(
                    "[pipecat-worker] WhatsApp reply wav audio track attached",
                    {
                        "callId": call_id,
                        "path": "reply.wav",
                    },
                    flush=True,
                )
                interrupted_text = await wait_for_reply_or_interruption(
                    call_id=call_id,
                    connection=connection_for_reply,
                    audio_track=audio_track,
                    reply_duration_seconds=reply_duration_seconds,
                    wav_path=wav_path,
                    sample_rate=sample_rate,
                    voice_threshold=voice_threshold,
                    max_voice_frames=max_voice_frames,
                    pre_speech_frames=pre_speech_frames,
                )
                if interrupted_text:
                    pending_user_text = interrupted_text
            else:
                print(
                    "[pipecat-worker] WhatsApp reply wav playback skipped",
                    {
                        "callId": call_id,
                        "reason": "missing_active_connection",
                    },
                    flush=True,
                )
                return

    except asyncio.CancelledError:
        print(
            "[pipecat-worker] WhatsApp audio input STT monitor cancelled",
            {"callId": call_id},
            flush=True,
        )
        raise
    except Exception as exc:
        print(
            "[pipecat-worker] WhatsApp audio input STT monitor failed",
            {
                "callId": call_id,
                "errorType": type(exc).__name__,
                "error": str(exc)[:500],
            },
            flush=True,
        )


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def get_safe_call_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    call_events = payload.get("callEvents")
    if not isinstance(call_events, list):
        return []

    safe_events: list[dict[str, Any]] = []

    for event in call_events:
        if not isinstance(event, dict):
            continue

        safe_events.append(
            {
                "callId": event.get("callId"),
                "event": event.get("event"),
                "direction": event.get("direction"),
                "phoneNumberId": event.get("phoneNumberId"),
                "fromMasked": mask_id(event.get("from")),
                "sdpPresent": bool(event.get("sdpPresent")),
            }
        )

    return safe_events


def send_whatsapp_text_message(*, to: str, text: str) -> dict[str, Any]:
    token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    if not token or not phone_number_id:
        return {
            "ok": False,
            "status": None,
            "bodyLength": 0,
            "error": "missing_whatsapp_env",
        }

    body = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text,
        },
    }

    encoded_body = json.dumps(body).encode("utf-8")
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/messages"

    req = urllib.request.Request(
        url,
        data=encoded_body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            response_body = res.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= res.status < 300,
                "status": res.status,
                "bodyLength": len(response_body),
                "bodyPreview": truncate_text(response_body),
            }
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "bodyLength": len(response_body),
            "bodyPreview": truncate_text(response_body),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "bodyLength": 0,
            "error": truncate_text(str(exc)),
        }


def execute_whatsapp_call_action(
    *,
    call_id: str,
    action: str,
    session: dict[str, Any] | None = None,
    to: str | None = None,
) -> dict[str, Any]:
    token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    if not token or not phone_number_id:
        return {
            "ok": False,
            "status": None,
            "bodyLength": 0,
            "error": "missing_whatsapp_env",
        }

    body: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "call_id": call_id,
        "action": action,
    }

    if to:
        body["to"] = to

    if session is not None:
        body["session"] = session

    encoded_body = json.dumps(body).encode("utf-8")
    url = f"https://graph.facebook.com/v25.0/{phone_number_id}/calls"

    req = urllib.request.Request(
        url,
        data=encoded_body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            response_body = res.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= res.status < 300,
                "status": res.status,
                "bodyLength": len(response_body),
                "bodyPreview": truncate_text(response_body),
            }
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "bodyLength": len(response_body),
            "bodyPreview": truncate_text(response_body),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "bodyLength": 0,
            "error": truncate_text(str(exc)),
        }


async def whatsapp_calls(request: web.Request) -> web.Response:
    expected_secret = os.getenv("WHATSAPP_CALL_WORKER_SECRET")

    if expected_secret:
        received_secret = request.headers.get("X-FitCoach-Call-Worker-Secret")
        if received_secret != expected_secret:
            return web.json_response({"received": False, "error": "unauthorized"}, status=401)

    payload = await request.json()
    safe_events = get_safe_call_events(payload)

    raw_calls = (
        payload.get("payload", {})
        .get("entry", [{}])[0]
        .get("changes", [{}])[0]
        .get("value", {})
        .get("calls", [])
    )

    raw_call = raw_calls[0] if isinstance(raw_calls, list) and raw_calls else {}
    raw_session = raw_call.get("session", {}) if isinstance(raw_call, dict) else {}
    raw_sdp = raw_session.get("sdp") if isinstance(raw_session, dict) else None
    raw_sdp_type = raw_session.get("sdp_type") if isinstance(raw_session, dict) else None
    raw_from = raw_call.get("from") if isinstance(raw_call, dict) else None

    print(
        "[pipecat-worker] WhatsApp calls received",
        {
            "source": payload.get("source"),
            "receivedAt": payload.get("receivedAt"),
            "events": safe_events,
        },
        flush=True,
    )

    terminate_events = [
        event
        for event in safe_events
        if event.get("event") == "terminate" and event.get("callId")
    ]

    for event in terminate_events:
        call_id = event.get("callId")
        task = ACTIVE_CALL_AUDIO_TASKS.pop(call_id, None)
        if task:
            task.cancel()

        connection = ACTIVE_CALL_CONNECTIONS.pop(call_id, None)
        if connection:
            await connection.cleanup()

        print(
            "[pipecat-worker] WhatsApp call resources cleaned",
            {
                "callId": call_id,
                "activeConnections": len(ACTIVE_CALL_CONNECTIONS),
                "activeAudioTasks": len(ACTIVE_CALL_AUDIO_TASKS),
            },
            flush=True,
        )

    connect_events = [
        event
        for event in safe_events
        if event.get("event") == "connect"
        and event.get("direction") == "USER_INITIATED"
        and event.get("callId")
    ]

    print(
        "[pipecat-worker] WhatsApp call action readiness",
        {
            "connectEventsDetected": len(connect_events),
            "whatsappEnvConfigured": bool(os.getenv("WHATSAPP_ACCESS_TOKEN"))
            and bool(os.getenv("WHATSAPP_PHONE_NUMBER_ID")),
        },
        flush=True,
    )

    handled = False

    for event in connect_events:
        call_id = event.get("callId")
        if not isinstance(call_id, str) or not call_id:
            continue

        if not isinstance(raw_sdp, str) or not raw_sdp:
            print(
                "[pipecat-worker] WhatsApp call pre_accept skipped",
                {
                    "callId": call_id,
                    "reason": "missing_sdp_offer",
                },
                flush=True,
            )
            continue

        connection = SmallWebRTCConnection()
        answer = None

        try:
            await connection.initialize(raw_sdp, raw_sdp_type or "offer")
            answer = connection.get_answer()
        except Exception as exc:
            print(
                "[pipecat-worker] WhatsApp call answer failed",
                {
                    "callId": call_id,
                    "errorType": type(exc).__name__,
                    "error": str(exc)[:300],
                },
                flush=True,
            )
            await connection.cleanup()
            continue

        if not isinstance(answer, dict) or not answer.get("sdp") or not answer.get("type"):
            print(
                "[pipecat-worker] WhatsApp call skipped",
                {
                    "callId": call_id,
                    "reason": "missing_sdp_answer",
                },
                flush=True,
            )
            await connection.cleanup()
            continue

        sdp_answer = filter_sdp_for_whatsapp(str(answer.get("sdp")))

        print(
            "[pipecat-worker] WhatsApp SDP answer ready",
            {
                "callId": call_id,
                "answerType": answer.get("type"),
                "answerSdpLength": len(sdp_answer),
                "pcIdPresent": bool(answer.get("pc_id")),
            },
            flush=True,
        )

        pre_accept_result = execute_whatsapp_call_action(
            call_id=call_id,
            action="pre_accept",
            to=raw_from if isinstance(raw_from, str) else None,
            session={
                "sdp_type": "answer",
                "sdp": sdp_answer,
            },
        )

        print(
            "[pipecat-worker] WhatsApp call pre_accept completed",
            {
                "callId": call_id,
                "ok": pre_accept_result.get("ok"),
                "status": pre_accept_result.get("status"),
                "bodyLength": pre_accept_result.get("bodyLength"),
                "bodyPreview": pre_accept_result.get("bodyPreview"),
                "error": pre_accept_result.get("error"),
            },
            flush=True,
        )

        if not pre_accept_result.get("ok"):
            await connection.cleanup()
            continue

        accept_result = execute_whatsapp_call_action(
            call_id=call_id,
            action="accept",
            to=raw_from if isinstance(raw_from, str) else None,
            session={
                "sdp_type": "answer",
                "sdp": sdp_answer,
            },
        )

        print(
            "[pipecat-worker] WhatsApp call accept completed",
            {
                "callId": call_id,
                "ok": accept_result.get("ok"),
                "status": accept_result.get("status"),
                "bodyLength": accept_result.get("bodyLength"),
                "bodyPreview": accept_result.get("bodyPreview"),
                "error": accept_result.get("error"),
            },
            flush=True,
        )

        if accept_result.get("ok"):
            hello_wav_path = ensure_hello_openai_wav()
            connection.replace_audio_track(WavAudioTrack(hello_wav_path))
            ACTIVE_CALL_CONNECTIONS[call_id] = connection
            ACTIVE_CALL_AUDIO_TASKS[call_id] = asyncio.create_task(
                monitor_audio_input(
                    call_id,
                    connection,
                    raw_from if isinstance(raw_from, str) else None,
                )
            )
            print(
                "[pipecat-worker] WhatsApp wav audio track attached",
                {
                    "callId": call_id,
                    "path": hello_wav_path,
                    "activeConnections": len(ACTIVE_CALL_CONNECTIONS),
                    "audioInputMonitorStarted": True,
                },
                flush=True,
            )
            handled = True
        else:
            await connection.cleanup()

    return web.json_response({"received": True, "handled": handled})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/whatsapp/calls", whatsapp_calls)
    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    web.run_app(create_app(), host="0.0.0.0", port=port)
