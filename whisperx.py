import os
import time
import tempfile
from faster_whisper import WhisperModel
from openai import OpenAI
from pydub import AudioSegment

# --- CONFIGURATION ---
DEFAULT_MODEL = "medium"
VERSION = "1.0"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
OPENAI_API_KEY = "YOUR_API_KEY_HERE"  # <-- Paste your OpenAI API key here (only needed for Diarize mode)
MAX_CHUNK_SECONDS = 1200  # OpenAI limit is 1500s; use 1200s (20 min) for safety margin
# ---------------------

LOGO = r"""
 __        ___     _                     __  __
 \ \      / / |__ (_)___ _ __   ___ _ __\ \/ /
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|\  /
   \ V  V / | | | | \__ \ |_) |  __/ |   /  \
    \_/\_/  |_| |_|_|___/ .__/ \___|_|  /_/\_\
                        |_|
"""

def main():
    print(LOGO)
    print(f"  Version {VERSION}")
    print("  Created by Jeppe Sverdrup")
    print()

    # Model selection
    print("Available models:")
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
    print("Model loaded. Ready to work.")

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

            # Load audio to check duration and split if needed
            audio = AudioSegment.from_file(user_input)
            audio_duration_s = len(audio) / 1000.0
            chunk_ms = MAX_CHUNK_SECONDS * 1000

            all_segments = []

            if audio_duration_s <= MAX_CHUNK_SECONDS:
                # Short enough — send directly
                print(f"Audio duration: {audio_duration_s:.0f}s (within API limit)")
                with open(user_input, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        file=audio_file,
                        model="gpt-4o-transcribe-diarize",
                        response_format="diarized_json",
                        chunking_strategy="auto",
                    )
                all_segments = transcript.to_dict().get("segments", [])
            else:
                # Too long — split into chunks
                num_chunks = -(-len(audio) // chunk_ms)  # ceiling division
                print(f"Audio duration: {audio_duration_s:.0f}s (exceeds API limit of {MAX_CHUNK_SECONDS}s)")
                print(f"Splitting into {num_chunks} chunks of ~{MAX_CHUNK_SECONDS}s each...")

                for i in range(num_chunks):
                    chunk_start_ms = i * chunk_ms
                    chunk_end_ms = min((i + 1) * chunk_ms, len(audio))
                    chunk = audio[chunk_start_ms:chunk_end_ms]
                    time_offset_s = chunk_start_ms / 1000.0

                    print(f"\n  Chunk {i + 1}/{num_chunks} [{chunk_start_ms / 1000:.0f}s - {chunk_end_ms / 1000:.0f}s]...", end=" ", flush=True)

                    # Export chunk to a temp WAV file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                        chunk.export(tmp_path, format="wav")

                    try:
                        with open(tmp_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                file=audio_file,
                                model="gpt-4o-transcribe-diarize",
                                response_format="diarized_json",
                                chunking_strategy="auto",
                            )

                        chunk_segments = transcript.to_dict().get("segments", [])

                        # Offset timestamps to reflect position in the full audio
                        for seg in chunk_segments:
                            seg["start"] = seg.get("start", 0) + time_offset_s
                            seg["end"] = seg.get("end", 0) + time_offset_s
                            all_segments.append(seg)

                        print(f"OK ({len(chunk_segments)} segments)")
                    finally:
                        os.remove(tmp_path)

                print()

            # Save the files
            file_base = f"{user_input}_diarize"
            file_log = file_base + "_log.txt"
            file_clean = file_base + "_clean.txt"

            with open(file_log, "w", encoding="utf-8") as f_log, \
                 open(file_clean, "w", encoding="utf-8") as f_clean:

                print("-" * 50)

                current_speaker = None

                for seg in all_segments:
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "").strip()
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)

                    # Write Log (Speaker + Time + Text)
                    timestamp = f"[{start:.2f}s -> {end:.2f}s]"
                    f_log.write(f"{speaker} {timestamp} {text}\n")

                    # Write Clean (Speaker: Text, grouped by speaker turns)
                    if speaker != current_speaker:
                        if current_speaker is not None:
                            f_clean.write("\n")
                        f_clean.write(f"{speaker}: ")
                        current_speaker = speaker
                    f_clean.write(text + " ")

                    # Live Preview
                    print(f"  {speaker} {timestamp} {text}")

            duration = time.time() - start_time
            print("-" * 50)
            print(f"DONE! ({duration:.1f}s)")
            print(f"1. Log:   {os.path.basename(file_log)}")
            print(f"2. Clean: {os.path.basename(file_clean)}")

        else:
            # --- Local Whisper transcribe/translate ---
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