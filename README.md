# WhisperX

```
 __        ___     _                     __  __
 \ \      / / |__ (_)___ _ __   ___ _ __\ \/ /
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|\  /
   \ V  V / | | | | \__ \ |_) |  __/ |   /  \
    \_/\_/  |_| |_|_|___/ .__/ \___|_|  /_/\_\
                        |_|
```

**A simple audio transcription tool with four modes: Transcribe, Translate, Diarize, and ElevenLabs Scribe.**

Created by Jeppe Sverdrup

---

## What It Does

WhisperX is a command-line tool that turns audio files into text. It runs locally on your machine using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (no cloud required for transcription and translation), and optionally connects to the OpenAI API for speaker diarization or the ElevenLabs API for advanced speech-to-text.

The local Whisper model is only loaded when you choose Transcribe or Translate, so API-only modes (`d`, `e`) start instantly without waiting for the model to load.

### Four Modes

| Mode | Key | Description |
|------|-----|-------------|
| **Transcribe** | `Enter` | Transcribes audio and keeps the original language |
| **Translate** | `t` | Transcribes audio and translates it into English |
| **Diarize** | `d` | Transcribes audio and identifies who said what (requires OpenAI API key) |
| **ElevenLabs Scribe** | `e` | Advanced cloud STT with diarization, PII redaction, and more (requires ElevenLabs API key) |

### Output Files

For every audio file processed, WhisperX saves two output files next to the input file:

- **`_log.txt`** — Full transcript with timestamps for every segment
- **`_clean.txt`** — Clean transcript with text only (no timestamps), easy to copy and paste

---

## Requirements

- Python 3.9+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [openai](https://pypi.org/project/openai/) Python SDK
- [pydub](https://pypi.org/project/pydub/)
- [requests](https://pypi.org/project/requests/)
- [ffmpeg](https://ffmpeg.org/) (required by pydub for audio processing)
- An OpenAI API key *(only required for Diarize mode)*
- An ElevenLabs API key *(only required for ElevenLabs Scribe mode)*

### Install dependencies

```bash
pip install faster-whisper openai pydub requests
```

> **ffmpeg** must also be installed and available on your system PATH. Download it from [ffmpeg.org](https://ffmpeg.org/download.html).

---

## Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/whisperx.git
   cd whisperx
   ```

2. **Install dependencies** (see above)

3. **Add your API key(s)**:
   Open `whisperx.py` and replace the placeholder(s) near the top of the file:
   ```python
   OPENAI_API_KEY = "YOUR_API_KEY_HERE"      # <-- needed for Diarize mode
   ELEVENLABS_API_KEY = "YOUR_API_KEY_HERE"  # <-- needed for ElevenLabs Scribe mode
   ```

---

## Usage

### Option A — Run directly with Python

```bash
python whisperx.py
```

### Option B — Windows launcher (double-click)

Edit `whisperx.bat` to point to your Python installation and script location, then double-click to launch.

> **Note:** The included `whisperx.bat` is configured for a specific local path (`C:\Whisper`). Edit the paths in the file to match your setup before using it.

### Workflow

1. Launch the tool
2. Drag and drop an audio file into the terminal window
3. Choose your mode: Transcribe, Translate, Diarize, or ElevenLabs Scribe
4. **Transcribe / Translate only:** Select a Whisper model (or press Enter for the default: `medium`). The model loads on first use and is cached for the rest of the session.
5. Output files are saved next to your audio file automatically

---

## Whisper Models

WhisperX supports all standard Whisper model sizes. Larger models are more accurate but slower and require more memory.

| Model | Size | Use case |
|-------|------|----------|
| `tiny` | ~39M params | Fast, lower accuracy |
| `base` | ~74M params | Good for quick jobs |
| `small` | ~244M params | Balanced |
| `medium` | ~769M params | **Default — good accuracy** |
| `large-v1/v2/v3` | ~1.5B params | Best accuracy, slowest |
| `large-v3-turbo` | ~809M params | Fast large model |

All models run **locally on CPU** — no internet connection required for Transcribe and Translate modes. The model is only loaded when you first choose Transcribe or Translate, so API-only modes (`d`, `e`) skip the model load entirely.

---

## ElevenLabs Scribe Mode (Advanced Cloud STT)

ElevenLabs Scribe mode sends your audio to the [ElevenLabs Speech-to-Text API](https://elevenlabs.io/docs/api-reference/speech-to-text) (`scribe_v2` model) and offers five sub-options:

| Option | Feature |
|--------|---------|
| **[1] Quick transcribe** | Auto-detect language, word timestamps, audio event tags |
| **[2] + Diarize** | Speaker identification (up to 32 speakers) |
| **[3] + Clean** | `no_verbatim` mode — strips filler words and false starts |
| **[4] + Redact PII** | Detects & masks names, SSNs, credit cards, medical data, and more |
| **[5] Custom** | Configure language, diarization, cleaning, events, PII, timestamp granularity, and key terms |

- Requires a valid ElevenLabs API key set in `whisperx.py`
- Uses raw HTTP requests (no ElevenLabs SDK needed)
- Output labels speakers as `speaker_0`, `speaker_1`, etc. (when diarization is enabled)

**Cost:** ElevenLabs Scribe mode makes API calls to ElevenLabs and will incur usage costs depending on audio length.

---

## Diarize Mode (Speaker Identification)

Diarize mode uses OpenAI's `gpt-4o-transcribe-diarize` API to identify and label individual speakers in the audio.

- Requires a valid OpenAI API key set in `whisperx.py`
- Audio longer than 20 minutes is automatically split into chunks and stitched back together
- Output labels speakers as `speaker_0`, `speaker_1`, etc.

**Cost:** Diarize mode makes API calls to OpenAI and will incur usage costs depending on audio length.

---

## Supported Audio Formats

Any format supported by ffmpeg, including:
- `.mp3`, `.mp4`, `.m4a`
- `.wav`, `.ogg`, `.flac`
- `.webm`, and more

---

## License

MIT License — free to use, modify, and distribute.
