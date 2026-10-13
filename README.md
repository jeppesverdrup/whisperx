# WhisperX

```
 __        ___     _                     __  __
 \ \      / / |__ (_)___ _ __   ___ _ __\ \/ /
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|\  /
   \ V  V / | | | | \__ \ |_) |  __/ |   /  \
    \_/\_/  |_| |_|_|___/ .__/ \___|_|  /_/\_\
                        |_|
```

**A simple audio transcription tool with five modes: Transcribe, Translate, Diarize, ElevenLabs Scribe, and GPT Transcribe.**

Created by Jeppe Sverdrup

---

## What It Does

WhisperX is a command-line tool that turns audio files into text. It runs locally on your machine using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (no cloud required for transcription and translation), and optionally connects to the OpenAI API for speaker diarization or cloud transcription, or the ElevenLabs API for advanced speech-to-text.

The local Whisper model is only loaded when you choose Transcribe or Translate, so API-only modes (`d`, `e`, `g`) start instantly without waiting for the model to load.

### Five Modes

| Mode | Key | Description |
|------|-----|-------------|
| **Transcribe** | `Enter` | Transcribes audio and keeps the original language |
| **Translate** | `t` | Transcribes audio and translates it into English |
| **Diarize** | `d` | Transcribes audio and identifies who said what (requires OpenAI API key) |
| **ElevenLabs Scribe** | `e` | Advanced cloud STT with diarization, PII redaction, and more (requires ElevenLabs API key) |
| **GPT Transcribe** | `g` | Cloud transcription with OpenAI `gpt-transcribe`, no speaker labels (requires OpenAI API key) |

### Output Files

For every audio file processed, WhisperX saves two output files next to the input file:

- **`_log.txt`** — Full transcript with timestamps for every segment (GPT Transcribe mode timestamps each ~2 minute chunk, since the model returns text only)
- **`_clean.txt`** — Clean transcript with text only (no timestamps), easy to copy and paste

---

## Requirements

- Python 3.9+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [openai](https://pypi.org/project/openai/) Python SDK
- [pydub](https://pypi.org/project/pydub/)
- [requests](https://pypi.org/project/requests/)
- [ffmpeg](https://ffmpeg.org/) (required by pydub for audio processing)
- An OpenAI API key *(only required for Diarize and GPT Transcribe modes)*
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
   OPENAI_API_KEY = "YOUR_API_KEY_HERE"      # <-- needed for Diarize and GPT Transcribe modes
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
3. Choose your mode: Transcribe, Translate, Diarize, ElevenLabs Scribe, or GPT Transcribe
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

All models run **locally on CPU** — no internet connection required for Transcribe and Translate modes. The model is only loaded when you first choose Transcribe or Translate, so API-only modes (`d`, `e`, `g`) skip the model load entirely.

---

## ElevenLabs Scribe Mode (Advanced Cloud STT)

ElevenLabs Scribe mode sends your audio to the [ElevenLabs Speech-to-Text API](https://elevenlabs.io/docs/api-reference/speech-to-text) (`scribe_v2` model) and offers two sub-options:

| Option | Feature |
|--------|---------|
| **[1] Fast track** | English transcription with diarization, auto speaker detection, filler removal, no audio event tags, no PII redaction, and word timestamps |
| **[2] Custom** | Configure model, language, timestamps, diarization, speaker roles, multi-channel transcription, cleaning, audio events, entity detection/redaction, key terms, temperature, seed, and logging |

- Requires a valid ElevenLabs API key set in `whisperx.py`
- Pressing Enter at the ElevenLabs submenu selects **Fast track**
- Uses raw HTTP requests (no ElevenLabs SDK needed)
- Output labels speakers as `speaker_0`, `speaker_1`, etc. (when diarization is enabled)

**Cost:** ElevenLabs Scribe mode makes API calls to ElevenLabs and will incur usage costs depending on audio length.

---

## Diarize Mode (Speaker Identification)

Diarize mode uses OpenAI's `gpt-4o-transcribe-diarize` API to identify and label individual speakers in the audio.

- Requires a valid OpenAI API key set in `whisperx.py`
- Audio is converted to compressed mono 16 kHz chunks (OpenAI limits uploads to 25 MB), split at quiet moments, and stitched back together
- Chunks are up to ~20 minutes; speaker labels are assigned per chunk, so they are not guaranteed to match across chunks
- Output labels unnamed speakers per chunk as `A`, `B`, `C`, ... and the same letter can mean different people in different chunks (use known speakers below to avoid this)
- **Known speakers (optional, up to 4):** give a name plus a 2-10 s sample of only that person, either `START-END` seconds in the recording (e.g. `36-44` or `1:05-1:12`) or a short audio file. That name is then used as the label across the whole file, in every chunk. Anyone without a sample still gets a per-chunk letter. Voice samples you reuse can be saved once in `KNOWN_SPEAKER_FILES` at the top of `whisperx.py`
- `_log.txt` keeps every segment. `_clean.txt` folds short interjections ("yeah", "mm-hmm") that sit inside someone else's turn into that turn and merges consecutive segments from the same speaker, so it reads as a conversation

**Cost:** Diarize mode makes API calls to OpenAI and will incur usage costs depending on audio length.

---

## GPT Transcribe Mode (Cloud STT)

GPT Transcribe mode sends your audio to OpenAI's `gpt-transcribe` model through the same transcription endpoint and API key as Diarize mode.

- Requires a valid OpenAI API key set in `whisperx.py`
- Audio is converted to compressed mono chunks of ~2 minutes, cut at quiet moments
- The model returns text only (no timestamps, no speakers), so `_log.txt` is timestamped per chunk and `_clean.txt` has one paragraph per chunk
- You are asked for **key terms** (names, product and jargon words, comma-separated). They are sent as `keywords` and make spellings consistent across chunks, e.g. so a name isn't written three different ways in one recording. Press Enter to skip
- Terms you always want can be set once in `GPT_TRANSCRIBE_KEYWORDS` at the top of `whisperx.py`; terms typed at the prompt are added to them
- Output files are named `<audio>_gpt-transcribe_log.txt` and `<audio>_gpt-transcribe_clean.txt`
- Use Diarize (`d`) when you need speaker labels

**Cost:** GPT Transcribe mode makes API calls to OpenAI and will incur usage costs depending on audio length.

---

## Supported Audio Formats

Any format supported by ffmpeg, including:
- `.mp3`, `.mp4`, `.m4a`
- `.wav`, `.ogg`, `.flac`
- `.webm`, and more

---

## License

MIT License — free to use, modify, and distribute.
