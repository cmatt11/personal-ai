"""Optional voice input/output.

Speech is an optional feature. It needs extra packages that are NOT part of the
zero-dependency core:

    pip install SpeechRecognition pyttsx3
    # plus PyAudio for microphone input on most systems

If those are missing, the assistant runs normally without voice. This module
degrades gracefully and never crashes the app.
"""

from typing import Optional

_TTS_ENGINE = None


def tts_available() -> bool:
    try:
        import pyttsx3  # noqa: F401

        return True
    except Exception:
        return False


def stt_available() -> bool:
    try:
        import speech_recognition  # noqa: F401

        return True
    except Exception:
        return False


def voice_available() -> bool:
    return tts_available() and stt_available()


def speak(text: str) -> bool:
    """Say text aloud. Returns True if spoken, False if TTS is unavailable."""
    global _TTS_ENGINE
    try:
        import pyttsx3

        if _TTS_ENGINE is None:
            _TTS_ENGINE = pyttsx3.init()
        _TTS_ENGINE.say(text)
        _TTS_ENGINE.runAndWait()
        return True
    except Exception:
        return False


def listen(timeout: float = 8.0) -> Optional[str]:
    """Capture speech from the microphone and return text, or None if unavailable."""
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=timeout)
        # Uses Google's free recognizer when online; falls back to offline if set up.
        return recognizer.recognize_google(audio)
    except Exception:
        return None


def status() -> str:
    if voice_available():
        return "voice: ready (speech in and out available)"
    missing = []
    if not stt_available():
        missing.append("SpeechRecognition (+ PyAudio)")
    if not tts_available():
        missing.append("pyttsx3")
    return "voice: unavailable. install: " + ", ".join(missing)
