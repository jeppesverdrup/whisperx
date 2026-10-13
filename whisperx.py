import os
import re
import io
import time
import base64
import tempfile
import subprocess
import requests
from faster_whisper import WhisperModel
from openai import OpenAI, OpenAIError
from pydub import AudioSegment

# --- CONFIGURATION ---
DEFAULT_MODEL = "medium"
VERSION = "1.6"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
OPENAI_API_KEY = "YOUR_API_KEY_HERE"  # <-- Paste your OpenAI API key here (needed for Diarize and GPT Transcribe modes)
ELEVENLABS_API_KEY = "YOUR_API_KEY_HERE"  # <-- Paste your ElevenLabs API key here (only needed for ElevenLabs mode)
MAX_CHUNK_SECONDS = 1200  # OpenAI limit is 1500s; use 1200s (20 min) for safety margin
GPT_TRANSCRIBE_CHUNK_SECONDS = 120  # gpt-transcribe returns no timestamps, so chunk length sets log granularity
KNOWN_SPEAKER_FILES = {}  # <-- Saved voice samples for Diarize (2-10 s each, max 4), e.g. {"Trond": r"C:\Whisper\voices\trond.wav"}
GPT_TRANSCRIBE_KEYWORDS = []  # <-- Names/terms always sent to GPT Transcribe, e.g. ["Trond", "EDW"]; more can be added per run
OPENAI_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # OpenAI audio upload limit
MAX_KNOWN_SPEAKERS = 4  # OpenAI limit for known speaker references
VOICE_SAMPLE_SECONDS = (2, 10)  # OpenAI requires each reference clip to be 2-10 s
BACKCHANNEL_WORDS = {"yeah", "yes", "yep", "mm", "mhm", "mm-hmm", "hmm", "uh-huh", "okay", "ok", "right", "exactly", "sure", "oh", "uh", "um", "absolutely"}
BACKCHANNEL_MAX_WORDS = 2  # interjections up to this length are folded into the surrounding turn in _clean.txt
OPENAI_CHUNK_BITRATE = "48k"  # mono 16 kHz speech; ~6 KB/s so a 20 min chunk is ~7 MB
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
# ---------------------

LOGO = r"""
 __        ___     _                     __  __
 \ \      / / |__ (_)___ _ __   ___ _ __\ \/ /
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|\  /
   \ V  V / | | | | \__ \ |_) |  __/ |   /  \
    \_/\_/  |_| |_|_|___/ .__/ \___|_|  /_/\_\
                        |_|
"""

def load_audio_mono(path):
    """Decode to mono 16 kHz inside ffmpeg so long stereo recordings stay small in memory."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [AudioSegment.converter, "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"ffmpeg could not decode the file: {result.stderr.strip()}")
        return AudioSegment.from_wav(tmp_path)
    finally:
        os.remove(tmp_path)


def quietest_offset_ms(window, step_ms=100):
    """Offset (ms) of the quietest slice in an audio window, used to cut between words."""
    best_offset, best_rms = 0, None
    for offset in range(0, max(len(window) - step_ms, 1), step_ms):
        rms = window[offset:offset + step_ms].rms
        if best_rms is None or rms < best_rms:
            best_offset, best_rms = offset, rms
    return best_offset + step_ms // 2


def plan_chunks(audio, target_s, search_s=15):
    """Split points as (start_ms, end_ms) pairs, cutting at the quietest moment near each target length."""
    target_ms = target_s * 1000
    search_ms = search_s * 1000
    bounds = []
    start = 0
    while len(audio) - start > target_ms + search_ms:
        window_start = start + target_ms - search_ms
        cut = window_start + quietest_offset_ms(audio[window_start:start + target_ms])
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


def iter_chunks(audio, bounds):
    """Yield (index, total, start_s, end_s, mp3_path) per chunk; the temp file is removed afterwards."""
    for i, (start_ms, end_ms) in enumerate(bounds):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            audio[start_ms:end_ms].export(tmp_path, format="mp3", bitrate=OPENAI_CHUNK_BITRATE)
            size = os.path.getsize(tmp_path)
            if size > OPENAI_MAX_UPLOAD_BYTES:
                raise ValueError(f"Chunk is {size / 1048576:.1f} MB, over the 25 MB upload limit.")
            yield i, len(bounds), start_ms / 1000.0, end_ms / 1000.0, tmp_path
        finally:
            os.remove(tmp_path)


def parse_time(text):
    """Seconds from '83', '83.5', '1:23' or '1:02:03'."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"'{text}' is not a time")
    seconds = 0.0
    for part in parts:
        try:
            seconds = seconds * 60 + float(part)
        except ValueError:
            raise ValueError(f"'{text}' is not a time") from None
    return seconds


def load_voice_sample(ref, audio):
    """Reference clip for one speaker: a short audio file, or START-END seconds inside this recording."""
    ref = ref.strip().strip('"')
    if os.path.isfile(ref):
        clip = load_audio_mono(ref)
    else:
        match = re.match(r"^\s*([\d:.]+)\s*[-\u2013]\s*([\d:.]+)\s*$", ref)
        if not match:
            raise ValueError("expected a file path or START-END in this recording, e.g. 36-44 or 1:05-1:12")
        start, end = parse_time(match.group(1)), parse_time(match.group(2))
        if end <= start or end * 1000 > len(audio):
            raise ValueError(f"{ref} is outside this {len(audio) / 1000:.0f}s recording")
        clip = audio[int(start * 1000):int(end * 1000)]
    seconds = len(clip) / 1000.0
    low, high = VOICE_SAMPLE_SECONDS
    if not low <= seconds <= high:
        raise ValueError(f"the sample is {seconds:.1f}s but must be {low}-{high}s")
    return clip


def clip_to_data_url(clip):
    buf = io.BytesIO()
    clip.export(buf, format="wav")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ask_known_speakers(audio):
    """Saved voice samples plus any added now. Returns (names, data_urls) for the diarize request."""
    names, refs = [], []

    for name, path in KNOWN_SPEAKER_FILES.items():
        if len(names) >= MAX_KNOWN_SPEAKERS:
            break
        try:
            refs.append(clip_to_data_url(load_voice_sample(path, audio)))
            names.append(name)
            print(f"Using saved voice sample for {name}.")
        except ValueError as e:
            print(f"Skipping saved voice sample for {name}: {e}")

    if len(names) < MAX_KNOWN_SPEAKERS:
        low, high = VOICE_SAMPLE_SECONDS
        print(f"\nKnown speakers (optional, up to {MAX_KNOWN_SPEAKERS}): give a name and a {low}-{high}s sample of only that person")
        print("  so their name is used as the label across the whole file. Press Enter to skip.")
        while len(names) < MAX_KNOWN_SPEAKERS:
            name = input("  Speaker name (Enter to finish): ").strip()
            if not name:
                break
            if name in names:
                print(f"  {name} is already added.")
                continue
            ref = input(f"  Sample for {name} (file path, or START-END in this recording e.g. 36-44 or 1:05-1:12): ")
            try:
                refs.append(clip_to_data_url(load_voice_sample(ref, audio)))
                names.append(name)
            except ValueError as e:
                print(f"  Not added: {e}")

    return names, refs


def tidy_turns(segments):
    """Speaker turns for _clean.txt: drop short 'yeah'/'mm-hmm' interjections that sit inside someone
    else's turn, drop empty segments, and join consecutive segments from the same speaker."""
    items = [(seg.get("speaker", "Unknown"), seg.get("text", "").strip()) for seg in segments]
    items = [(speaker, text) for speaker, text in items if text]

    def is_backchannel(text):
        words = re.findall(r"[\w'-]+", text.lower())
        return 0 < len(words) <= BACKCHANNEL_MAX_WORDS and all(w in BACKCHANNEL_WORDS for w in words)

    backchannel = [is_backchannel(text) for _, text in items]

    # nearest non-interjection neighbour on each side of every item
    prev_real, last = [], None
    for i in range(len(items)):
        prev_real.append(last)
        if not backchannel[i]:
            last = i
    next_real, nxt = [None] * len(items), None
    for i in range(len(items) - 1, -1, -1):
        next_real[i] = nxt
        if not backchannel[i]:
            nxt = i

    turns = []
    for i, (speaker, text) in enumerate(items):
        if backchannel[i]:
            p, n = prev_real[i], next_real[i]
            if p is not None and n is not None and items[p][0] == items[n][0] != speaker:
                continue
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += " " + text
        else:
            turns.append([speaker, text])
    return [(speaker, text) for speaker, text in turns]


def main():
    print(LOGO)
    print(f"  Version {VERSION}")
    print("  Created by Jeppe Sverdrup")
    print()

    model = None
    model_size = None

    while True:
        # 1. Get the file
        user_input = input("\nDrag and drop an audio file here (or type 'exit'): ").strip().strip('"')
        
        if user_input.lower() in ["exit", "quit"]:
            break

        if not os.path.exists(user_input):
            print("Error: File not found!")
            continue

        # 2. The Toggle: Ask for mode
        print("\nChoose Mode:")
        print("   [Enter] = Transcribe (Keep original language)")
        print("   [  t  ] = Translate (Audio -> English Text)")
        print("   [  d  ] = Diarize (Identify speakers via OpenAI API)")
        print("   [  e  ] = ElevenLabs Scribe (Advanced STT via ElevenLabs API)")
        print("   [  g  ] = GPT Transcribe (Cloud STT via OpenAI API, no speakers)")
        mode_input = input("Selection: ").strip().lower()

        if mode_input == 'd':
            # --- Diarization via OpenAI API ---
            if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
                print("\nError: OpenAI API key not configured!")
                print("Set OPENAI_API_KEY in whisperx.py to use diarization.")
                continue

            print(f"\n---> Diarizing '{os.path.basename(user_input)}' via OpenAI API...")
            start_time = time.time()

            client = OpenAI(api_key=OPENAI_API_KEY)

            # Decode, then split at quiet points into compressed chunks that fit the upload limit
            try:
                audio = load_audio_mono(user_input)
            except ValueError as e:
                print(f"\nError: {e}")
                continue
            audio_duration_s = len(audio) / 1000.0
            bounds = plan_chunks(audio, MAX_CHUNK_SECONDS)

            speaker_kwargs = {}
            known_names, known_refs = ask_known_speakers(audio)
            if known_names:
                speaker_kwargs = {"known_speaker_names": known_names, "known_speaker_references": known_refs}
            start_time = time.time()

            if len(bounds) == 1:
                print(f"Audio duration: {audio_duration_s:.0f}s (within API limit)")
            else:
                print(f"Audio duration: {audio_duration_s:.0f}s (exceeds API limit of {MAX_CHUNK_SECONDS}s)")
                print(f"Splitting into {len(bounds)} chunks of ~{MAX_CHUNK_SECONDS}s each...")

            all_segments = []

            try:
                for i, total, chunk_start_s, chunk_end_s, tmp_path in iter_chunks(audio, bounds):
                    if total > 1:
                        print(f"\n  Chunk {i + 1}/{total} [{chunk_start_s:.0f}s - {chunk_end_s:.0f}s]...", end=" ", flush=True)

                    with open(tmp_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            file=audio_file,
                            model="gpt-4o-transcribe-diarize",
                            response_format="diarized_json",
                            chunking_strategy="auto",
                            **speaker_kwargs,
                        )

                    chunk_segments = transcript.to_dict().get("segments", [])

                    # Offset timestamps to reflect position in the full audio
                    for seg in chunk_segments:
                        seg["start"] = seg.get("start", 0) + chunk_start_s
                        seg["end"] = seg.get("end", 0) + chunk_start_s
                        all_segments.append(seg)

                    if total > 1:
                        print(f"OK ({len(chunk_segments)} segments)")
            except (OpenAIError, ValueError) as e:
                print(f"\nError from OpenAI API: {e}")
                continue

            print()

            # Save the files
            file_base = f"{user_input}_diarize"
            file_log = file_base + "_log.txt"
            file_clean = file_base + "_clean.txt"

            with open(file_log, "w", encoding="utf-8") as f_log, \
                 open(file_clean, "w", encoding="utf-8") as f_clean:

                print("-" * 50)

                for seg in all_segments:
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "").strip()
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)

                    # Write Log (Speaker + Time + Text), every segment
                    timestamp = f"[{start:.2f}s -> {end:.2f}s]"
                    f_log.write(f"{speaker} {timestamp} {text}\n")

                    # Live Preview
                    print(f"  {speaker} {timestamp} {text}")

                # Write Clean (Speaker: Text), interjections folded into the surrounding turn
                turns = tidy_turns(all_segments)
                f_clean.write("\n\n".join(f"{speaker}: {text}" for speaker, text in turns) + "\n")

            duration = time.time() - start_time
            print("-" * 50)
            print(f"DONE! ({duration:.1f}s)")
            print(f"1. Log:   {os.path.basename(file_log)}")
            print(f"2. Clean: {os.path.basename(file_clean)}")

        elif mode_input == 'g':
            # --- GPT Transcribe via OpenAI API ---
            if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
                print("\nError: OpenAI API key not configured!")
                print("Set OPENAI_API_KEY in whisperx.py to use GPT Transcribe.")
                continue

            # Key terms keep names and jargon spelled the same in every chunk
            keywords = list(GPT_TRANSCRIBE_KEYWORDS)
            default_note = f"Enter to use the {len(keywords)} default(s)" if keywords else "Enter to skip"
            terms = input(f"\nKey terms/names to boost (comma-separated, {default_note}): ").strip()
            keywords += [t.strip() for t in terms.split(",") if t.strip()]
            extra_body = {"keywords": keywords} if keywords else None

            print(f"\n---> Transcribing '{os.path.basename(user_input)}' via OpenAI gpt-transcribe...")
            if keywords:
                print(f"Using {len(keywords)} key term(s).")
            start_time = time.time()

            client = OpenAI(api_key=OPENAI_API_KEY)

            try:
                audio = load_audio_mono(user_input)
            except ValueError as e:
                print(f"\nError: {e}")
                continue
            bounds = plan_chunks(audio, GPT_TRANSCRIBE_CHUNK_SECONDS)
            print(f"Audio duration: {len(audio) / 1000.0:.0f}s, sending {len(bounds)} chunk(s) of ~{GPT_TRANSCRIBE_CHUNK_SECONDS}s...")

            # gpt-transcribe returns text only (no timestamps), so the log is timestamped per chunk
            chunk_results = []
            languages = []

            try:
                for i, total, chunk_start_s, chunk_end_s, tmp_path in iter_chunks(audio, bounds):
                    print(f"  Chunk {i + 1}/{total} [{chunk_start_s:.0f}s - {chunk_end_s:.0f}s]...", end=" ", flush=True)

                    with open(tmp_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            file=audio_file,
                            model="gpt-transcribe",
                            response_format="json",
                            extra_body=extra_body,
                        )

                    result = transcript.to_dict()
                    chunk_results.append((chunk_start_s, chunk_end_s, result.get("text", "").strip()))
                    for lang in result.get("languages") or []:
                        code = lang.get("code")
                        if code and code not in languages:
                            languages.append(code)
                    print("OK")
            except (OpenAIError, ValueError) as e:
                print(f"\nError from OpenAI API: {e}")
                continue

            file_base = f"{user_input}_gpt-transcribe"
            file_log = file_base + "_log.txt"
            file_clean = file_base + "_clean.txt"

            with open(file_log, "w", encoding="utf-8") as f_log, \
                 open(file_clean, "w", encoding="utf-8") as f_clean:

                if languages:
                    print(f"Detected language(s): {', '.join(code.upper() for code in languages)}")
                print("-" * 50)

                for start, end, text in chunk_results:
                    timestamp = f"[{start:.2f}s -> {end:.2f}s]"
                    f_log.write(f"{timestamp} {text}\n")
                    print(f"  {timestamp} {text}")

                # One paragraph per chunk so a long recording is scannable
                f_clean.write("\n\n".join(text for _, _, text in chunk_results if text) + "\n")

            duration = time.time() - start_time
            print("-" * 50)
            print(f"DONE! ({duration:.1f}s)")
            print(f"1. Log:   {os.path.basename(file_log)}")
            print(f"2. Clean: {os.path.basename(file_clean)}")

        elif mode_input == 'e':
            # --- ElevenLabs Scribe STT ---
            if ELEVENLABS_API_KEY == "YOUR_API_KEY_HERE":
                print("\nError: ElevenLabs API key not configured!")
                print("Set ELEVENLABS_API_KEY in whisperx.py to use ElevenLabs mode.")
                continue

            print("\nElevenLabs Scribe Options:")
            print("   [1] Fast track (English, diarize, clean, word timestamps)")
            print("   [2] Custom (configure all available Scribe options)")
            el_choice = input("Selection [1-2, Enter=1]: ").strip()
            if el_choice == "":
                el_choice = "1"

            # Defaults
            el_model_id = "scribe_v2"
            el_diarize = False
            el_num_speakers = None
            el_diarization_threshold = None
            el_tag_events = True
            el_no_verbatim = False
            el_entity_detection = None
            el_entity_redaction = None
            el_entity_redaction_mode = None
            el_language = None
            el_keyterms = None
            el_timestamps = "word"
            el_temperature = None
            el_seed = None
            el_use_multi_channel = False
            el_detect_speaker_roles = False
            el_enable_logging = True

            if el_choice == "1":
                el_language = "en"
                el_diarize = True
                el_no_verbatim = True
                el_tag_events = False
                el_timestamps = "word"
                print("Fast track enabled: English, diarization with auto speakers, filler removal, no event tags, word timestamps.")
            elif el_choice == "2":
                # Custom configuration
                model_choice = input("Model ID (scribe_v2/scribe_v1) [scribe_v2]: ").strip().lower()
                if model_choice in ("scribe_v1", "scribe_v2"):
                    el_model_id = model_choice

                lang = input("Language code (e.g. 'en', 'no', or Enter for auto-detect): ").strip()
                if lang:
                    el_language = lang

                ts = input("Timestamp granularity (word/character/none) [word]: ").strip().lower()
                if ts in ("character", "none"):
                    el_timestamps = ts

                if input("Enable diarization? (y/N): ").strip().lower() == 'y':
                    el_diarize = True
                    speakers = input("  Max speakers (Enter for auto): ").strip()
                    if speakers == "":
                        pass
                    elif speakers.isdigit() and 1 <= int(speakers) <= 32:
                        el_num_speakers = int(speakers)
                    else:
                        threshold = input("  Diarization threshold 0.1-0.4 (Enter for API default): ").strip()
                        try:
                            threshold_value = float(threshold)
                            if 0.1 <= threshold_value <= 0.4:
                                el_diarization_threshold = threshold_value
                        except ValueError:
                            pass

                    if input("  Detect speaker roles, agent/customer? (y/N): ").strip().lower() == 'y':
                        el_detect_speaker_roles = True

                if input("Use multi-channel transcription? (y/N): ").strip().lower() == 'y':
                    el_use_multi_channel = True
                    if el_detect_speaker_roles:
                        print("Speaker-role detection is not available with multi-channel; disabling speaker roles.")
                        el_detect_speaker_roles = False

                if el_model_id == "scribe_v2" and input("Remove filler words? (y/N): ").strip().lower() == 'y':
                    el_no_verbatim = True

                if input("Tag audio events (laughter, applause, etc.)? (Y/n): ").strip().lower() == 'n':
                    el_tag_events = False

                entity_detection = input("Entity detection (all/pii/phi/pci/other/offensive_language, or Enter for none): ").strip()
                if entity_detection:
                    el_entity_detection = entity_detection
                    entity_redaction = input("Entity redaction (same scope, or Enter for detect only): ").strip()
                    if entity_redaction:
                        el_entity_redaction = entity_redaction
                        redaction_mode = input("Redaction mode (redacted/entity_type/enumerated_entity_type) [enumerated_entity_type]: ").strip()
                        if redaction_mode in ("redacted", "entity_type", "enumerated_entity_type"):
                            el_entity_redaction_mode = redaction_mode

                terms = input("Key terms to boost (comma-separated, or Enter to skip): ").strip()
                if terms:
                    el_keyterms = [t.strip() for t in terms.split(",") if t.strip()][:1000]

                temperature = input("Temperature 0-2 (Enter for API default): ").strip()
                try:
                    temperature_value = float(temperature)
                    if 0 <= temperature_value <= 2:
                        el_temperature = temperature_value
                except ValueError:
                    pass

                seed = input("Seed 0-2147483647 (Enter for none): ").strip()
                if seed.isdigit() and 0 <= int(seed) <= 2147483647:
                    el_seed = int(seed)

                if input("Enable ElevenLabs logging/storage? (Y/n): ").strip().lower() == 'n':
                    el_enable_logging = False
            else:
                print("Invalid selection. Using fast track.")
                el_language = "en"
                el_diarize = True
                el_no_verbatim = True
                el_tag_events = False
                el_timestamps = "word"

            # Build the request
            print(f"\n---> Transcribing '{os.path.basename(user_input)}' via ElevenLabs Scribe...")
            start_time = time.time()

            headers = {"xi-api-key": ELEVENLABS_API_KEY}
            data = {"model_id": el_model_id}

            if el_diarize:
                data["diarize"] = "true"
            if el_num_speakers:
                data["num_speakers"] = str(el_num_speakers)
            if el_diarization_threshold is not None:
                data["diarization_threshold"] = str(el_diarization_threshold)
            if el_tag_events:
                data["tag_audio_events"] = "true"
            else:
                data["tag_audio_events"] = "false"
            if el_no_verbatim:
                data["no_verbatim"] = "true"
            if el_language:
                data["language_code"] = el_language
            if el_timestamps:
                data["timestamps_granularity"] = el_timestamps
            if el_entity_detection:
                data["entity_detection"] = el_entity_detection
            if el_entity_redaction:
                data["entity_redaction"] = el_entity_redaction
            if el_entity_redaction_mode:
                data["entity_redaction_mode"] = el_entity_redaction_mode
            if el_temperature is not None:
                data["temperature"] = str(el_temperature)
            if el_seed is not None:
                data["seed"] = str(el_seed)
            if el_use_multi_channel:
                data["use_multi_channel"] = "true"
            if el_detect_speaker_roles:
                data["detect_speaker_roles"] = "true"
            if el_keyterms:
                # Use list of tuples to support multiple values for the same key
                data_tuples = list(data.items())
                for term in el_keyterms:
                    data_tuples.append(("keyterms", term))
                data = data_tuples

            with open(user_input, "rb") as audio_file:
                files = {"file": (os.path.basename(user_input), audio_file)}
                params = {"enable_logging": str(el_enable_logging).lower()}
                response = requests.post(ELEVENLABS_STT_URL, headers=headers, params=params, data=data, files=files)

            if response.status_code != 200:
                print(f"\nError from ElevenLabs API ({response.status_code}):")
                print(response.text)
                continue

            result = response.json()

            transcripts = result.get("transcripts")
            if transcripts:
                transcript_items = transcripts.items() if isinstance(transcripts, dict) else enumerate(transcripts)
                channel_texts = []
                combined_words = []
                combined_entities = []
                language = "unknown"
                lang_prob = 0

                for channel, transcript in transcript_items:
                    channel_label = f"channel_{channel}"
                    channel_text = transcript.get("text", "")
                    if channel_text:
                        channel_texts.append(f"{channel_label}: {channel_text}")

                    if language == "unknown":
                        language = transcript.get("language_code", "unknown")
                        lang_prob = transcript.get("language_probability", 0)

                    for word in transcript.get("words", []):
                        word["channel_index"] = channel
                        combined_words.append(word)

                    combined_entities.extend(transcript.get("entities", []))

                combined_words.sort(key=lambda word: (word.get("start", 0), word.get("end", 0)))
                result["language_code"] = language
                result["language_probability"] = lang_prob
                result["text"] = "\n\n".join(channel_texts)
                result["words"] = combined_words
                result["entities"] = combined_entities

            # Extract data
            language = result.get("language_code", "unknown")
            lang_prob = result.get("language_probability", 0)
            text = result.get("text", "")
            words = result.get("words", [])
            entities = result.get("entities", [])

            # Determine output suffix
            suffix_parts = ["elevenlabs"]
            if el_diarize:
                suffix_parts.append("diarize")
            if el_no_verbatim:
                suffix_parts.append("clean")
            if el_entity_redaction:
                suffix_parts.append("redacted")
            file_base = f"{user_input}_{'_'.join(suffix_parts)}"
            file_log = file_base + "_log.txt"
            file_clean = file_base + "_clean.txt"

            with open(file_log, "w", encoding="utf-8") as f_log, \
                 open(file_clean, "w", encoding="utf-8") as f_clean:

                print(f"Detected language: {language.upper()} (Confidence: {lang_prob:.2f})")
                print("-" * 50)

                if el_diarize and words:
                    # Group words by speaker turns
                    current_speaker = None
                    current_start = 0
                    current_words = []

                    def flush_speaker(speaker, start, end, word_list, f_log, f_clean, current_speaker_ref):
                        line_text = " ".join(w["text"] for w in word_list if w.get("type") != "spacing").strip()
                        if not line_text:
                            return
                        timestamp = f"[{start:.2f}s -> {end:.2f}s]"
                        f_log.write(f"{speaker} {timestamp} {line_text}\n")
                        if speaker != current_speaker_ref[0]:
                            if current_speaker_ref[0] is not None:
                                f_clean.write("\n")
                            f_clean.write(f"{speaker}: ")
                            current_speaker_ref[0] = speaker
                        f_clean.write(line_text + " ")
                        print(f"  {speaker} {timestamp} {line_text}")

                    speaker_ref = [None]
                    seg_speaker = None
                    seg_start = 0
                    seg_words = []

                    for w in words:
                        w_speaker = w.get("speaker_id", "Unknown")
                        if w_speaker != seg_speaker and seg_words:
                            seg_end = seg_words[-1].get("end", 0)
                            flush_speaker(seg_speaker, seg_start, seg_end, seg_words, f_log, f_clean, speaker_ref)
                            seg_words = []
                            seg_start = w.get("start", 0)
                        if not seg_words:
                            seg_start = w.get("start", 0)
                        seg_speaker = w_speaker
                        seg_words.append(w)

                    if seg_words:
                        seg_end = seg_words[-1].get("end", 0)
                        flush_speaker(seg_speaker, seg_start, seg_end, seg_words, f_log, f_clean, speaker_ref)

                elif words:
                    # No diarization — output word-level timestamps
                    for w in words:
                        w_type = w.get("type", "word")
                        w_text = w.get("text", "")
                        if w_type == "spacing":
                            continue
                        start = w.get("start", 0)
                        end = w.get("end", 0)
                        if w_type == "audio_event":
                            f_log.write(f"[{start:.2f}s -> {end:.2f}s] {w_text}\n")
                            print(f"  {w_text}")
                        else:
                            f_log.write(f"[{start:.2f}s -> {end:.2f}s] {w_text}\n")

                    # Write full text to clean file and preview
                    f_clean.write(text)
                    # Print sentence-level preview
                    for line in text.split(". "):
                        line = line.strip()
                        if line:
                            print(f"  {line}")

                else:
                    f_clean.write(text)
                    print(text)

                # Show detected entities if any
                if entities:
                    print(f"\n  Entities detected: {len(entities)}")
                    for ent in entities[:10]:
                        print(f"    [{ent.get('entity_type', '?')}] \"{ent.get('text', '')}\"")
                    if len(entities) > 10:
                        print(f"    ... and {len(entities) - 10} more")

            duration = time.time() - start_time
            print("-" * 50)
            print(f"DONE! ({duration:.1f}s)")
            print(f"1. Log:   {os.path.basename(file_log)}")
            print(f"2. Clean: {os.path.basename(file_clean)}")

        else:
            # --- Local Whisper transcribe/translate ---
            # Load model on first use
            if model is None:
                print("\nAvailable models:")
                for i, name in enumerate(AVAILABLE_MODELS, 1):
                    default_tag = " (default)" if name == DEFAULT_MODEL else ""
                    print(f"  [{i}] {name}{default_tag}")

                model_input = input(f"\nSelect model [1-{len(AVAILABLE_MODELS)}] or Enter for default: ").strip()

                if model_input == "":
                    model_size = DEFAULT_MODEL
                elif model_input.isdigit() and 1 <= int(model_input) <= len(AVAILABLE_MODELS):
                    model_size = AVAILABLE_MODELS[int(model_input) - 1]
                else:
                    print(f"Invalid selection. Using default ({DEFAULT_MODEL}).")
                    model_size = DEFAULT_MODEL

                print(f"\n--- Loading {model_size} Model (CPU) ---")
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                print("Model loaded.\n")

            if mode_input == 't':
                current_task = "translate"
                print(f"\n---> Translating '{os.path.basename(user_input)}' into ENGLISH...")
            else:
                current_task = "transcribe"
                print(f"\n---> Transcribing '{os.path.basename(user_input)}' in ORIGINAL language...")

            start_time = time.time()

            # The Work
            # We allow auto-detection of language by not setting 'language=' explicitly
            segments, info = model.transcribe(
                user_input,
                beam_size=5,
                task=current_task
            )

            # Save the Files
            file_base = f"{user_input}_{current_task}"
            file_log = file_base + "_log.txt"
            file_clean = file_base + "_clean.txt"

            with open(file_log, "w", encoding="utf-8") as f_log, \
                 open(file_clean, "w", encoding="utf-8") as f_clean:

                print(f"Detected language: {info.language.upper()} (Confidence: {info.language_probability:.2f})")
                print("-" * 50)

                for segment in segments:
                    text = segment.text.strip()

                    # Write Log (Time + Text)
                    timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
                    f_log.write(f"{timestamp} {text}\n")

                    # Write Clean (Text only)
                    f_clean.write(text + " ")

                    # Live Preview
                    print(text)

            duration = time.time() - start_time
            print("-" * 50)
            print(f"DONE! ({duration:.1f}s)")
            print(f"1. Log:   {os.path.basename(file_log)}")
            print(f"2. Clean: {os.path.basename(file_clean)}")

if __name__ == "__main__":
    main()
