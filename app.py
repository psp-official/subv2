import os
import re
import json
import tempfile
import subprocess
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Maximum upload size: 1 GB
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

UPLOAD_DIR = Path(tempfile.gettempdir()) / "movie_subtitle_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

# Change this model ID only if Google AI Studio shows a different
# supported model ID for your API key.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Tune these in Render Environment Variables without changing the code.
GEMINI_BATCH_SIZE = max(1, min(40, int(os.getenv("GEMINI_BATCH_SIZE", "25"))))
GEMINI_RETRIES = max(0, min(5, int(os.getenv("GEMINI_RETRIES", "3"))))
# Positive value delays subtitles slightly to compensate for early Whisper segment starts.
SUBTITLE_DELAY_SECONDS = float(os.getenv("SUBTITLE_DELAY_SECONDS", "0.0"))
TRANSLATION_CACHE_MAX = 5000
# In-memory cache: avoids repeat Gemini calls within the same running worker.
# It clears when the service restarts and is not shared between workers.
TRANSLATION_CACHE = OrderedDict()


def get_field(obj, name, default=None):
    """Read fields from dictionaries and API response objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def make_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    milliseconds = int(round(max(0.0, float(seconds)) * 1000))

    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)

    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{seconds:02},"
        f"{milliseconds:03}"
    )


def format_subtitle(text, max_chars=22):
    """Wrap Burmese subtitle text into at most two readable lines."""
    text = re.sub(r"\s+", " ", str(text)).strip()

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    # Burmese does not normally separate every word with spaces.
    # Prefer existing spaces as line-break points.
    words = text.split(" ")

    if len(words) == 1:
        # Avoid cutting Burmese combining characters away from
        # the base character.
        parts = re.findall(r".{1,22}", text)
        return "\n".join(parts[:2]) if len(parts) <= 2 else (
            parts[0] + "\n" + "".join(parts[1:])
        )

    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    if len(lines) <= 2:
        return "\n".join(lines)

    # Keep the full text rather than silently dropping words.
    first = lines[0]
    remaining = " ".join(lines[1:])
    return first + "\n" + remaining


def clean_json_response(text):
    """Remove Markdown fences around JSON."""
    text = str(text or "").strip()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE
    )
    return text.strip()


def _cache_get(text):
    key = re.sub(r"\s+", " ", str(text or "")).strip()
    value = TRANSLATION_CACHE.get(key)
    if value is not None:
        TRANSLATION_CACHE.move_to_end(key)
    return key, value


def _cache_put(key, value):
    TRANSLATION_CACHE[key] = value
    TRANSLATION_CACHE.move_to_end(key)
    while len(TRANSLATION_CACHE) > TRANSLATION_CACHE_MAX:
        TRANSLATION_CACHE.popitem(last=False)


def translate_batch(model, segments):
    """Translate only uncached Chinese dialogue into natural spoken Burmese."""
    translations = {}
    uncached = []
    cache_keys = {}

    for idx, segment in enumerate(segments):
        key, cached = _cache_get(segment["text"])
        cache_keys[idx] = key
        if cached is not None:
            translations[idx] = cached
        else:
            uncached.append({"id": idx, "text": segment["text"]})

    if uncached:
        prompt = f"""
You are a professional Chinese-to-Myanmar subtitle translator for Chinese movies.
Translate each Chinese dialogue line into natural, fluent, conversational Burmese
(Myanmar language) that sounds like real people speaking, not a literal machine translation.

RULES:
- Output Burmese only for each translation; do not leave the Chinese untranslated.
- Preserve the exact meaning, character intent, emotion, relationship, and tone.
- Use natural spoken Burmese appropriate to the scene; avoid stiff, bookish wording.
- Keep each subtitle concise enough to read on screen, but never omit important meaning.
- Do not summarize, explain, add narration, or invent details.
- Keep IDs exactly as supplied and return every item in the same order.
- If the Chinese is ambiguous, choose the most contextually safe translation.
- Return ONLY a valid JSON array with each object containing integer "id" and string "myanmar".

Example:
[{{"id": 0, "myanmar": "မင်း ဘာလုပ်နေတာလဲ။"}}]

Input:
{json.dumps(uncached, ensure_ascii=False)}
"""

        response = None
        for attempt in range(GEMINI_RETRIES + 1):
            try:
                response = model.generate_content(prompt)
                break
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = any(token in message for token in (
                    "429", "rate limit", "quota", "resource_exhausted",
                    "too many requests"
                ))
                if not is_rate_limit or attempt >= GEMINI_RETRIES:
                    raise
                # Back off on quota/rate-limit errors; do not retry other failures.
                delay = min(30, 2 ** attempt * 2)
                app.logger.warning(
                    "Gemini rate limit on attempt %s; retrying in %s seconds",
                    attempt + 1, delay
                )
                time.sleep(delay)

        response_text = get_field(response, "text", "")
        if not response_text:
            raise ValueError("Gemini returned an empty response.")

        result = json.loads(clean_json_response(response_text))
        if not isinstance(result, list):
            raise ValueError("Gemini response was not a JSON array.")

        returned = {}
        for item in result:
            if not isinstance(item, dict) or "id" not in item or "myanmar" not in item:
                continue
            try:
                idx = int(item["id"])
                translated = str(item["myanmar"]).strip()
                if translated:
                    returned[idx] = translated
            except (TypeError, ValueError):
                continue

        missing = [item["id"] for item in uncached if item["id"] not in returned]
        if missing:
            raise ValueError(f"Gemini did not translate all lines. Missing IDs: {missing}")

        for item in uncached:
            idx = item["id"]
            translations[idx] = returned[idx]
            _cache_put(cache_keys[idx], returned[idx])

    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    for idx, translated in translations.items():
        if chinese_pattern.search(translated):
            app.logger.warning("Translation for line %s still contains Chinese: %s", idx, translated)

    return [translations[i] for i in range(len(segments))]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/process")
def process_video():
    if "video" not in request.files:
        return jsonify(error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"), 400

    video = request.files["video"]

    try:
        user_delay = float(request.form.get("delay", 0.0))
    except ValueError:
        user_delay = 0.0

    if not video.filename:
        return jsonify(error="ဗီဒီယိုဖိုင်ကို အရင်ရွေးပါ။"), 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify(
            error="MP4, MOV, MKV သို့မဟုတ် WEBM ဖိုင်ကို ရွေးပါ။"
        ), 400

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not groq_key or not gemini_key:
        return jsonify(
            error=(
                "Render Environment Variables ထဲမှာ "
                "GROQ_API_KEY နဲ့ GEMINI_API_KEY ထည့်ပါ။"
            )
        ), 500

    try:
        from groq import Groq
        import google.generativeai as genai
    except ImportError:
        app.logger.exception("Required library import failed")
        return jsonify(
            error="လိုအပ်သော Python libraries များ မတပ်ဆင်ရသေးပါ။"
        ), 500

    job_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR))
    video_path = job_dir / filename
    audio_path = job_dir / "audio.mp3"

    try:
        video.save(video_path)

        # Extract audio only. Do not modify or export the video.
        command = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            str(audio_path)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode != 0 or not audio_path.exists():
            app.logger.error(
                "FFmpeg failed: %s",
                result.stderr[-3000:]
            )
            return jsonify(
                error="ဗီဒီယိုထဲက အသံကို ထုတ်ယူလို့မရပါ။"
            ), 400

        if audio_path.stat().st_size == 0:
            return jsonify(error="ဗီဒီယိုထဲမှာ အသံမတွေ့ပါ။"), 400

        # Transcribe Chinese speech with segment timestamps.
        groq_client = Groq(api_key=groq_key)

        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model="whisper-large-v3",
                language="zh",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )

        transcription_segments = get_field(
            transcription, "segments", []
        ) or []
        # Word-level timestamps let each subtitle start at the first detected
        # spoken word inside its Whisper segment, instead of using a fixed
        # offset for every movie. Fall back to segment timestamps if absent.
        transcription_words = get_field(transcription, "words", []) or []
        # Some transcription responses expose word timestamps under each
        # segment rather than at the top level. Collect both response shapes.
        if not transcription_words:
            nested_words = []
            for _segment in transcription_segments:
                nested_words.extend(
                    get_field(_segment, "words", []) or []
                )
            transcription_words = nested_words

        raw_segments = []

        for segment in transcription_segments:
            try:
                text = str(
                    get_field(segment, "text", "") or ""
                ).strip()

                start = float(
                    get_field(segment, "start", 0) or 0
                )

                end = float(
                    get_field(segment, "end", start) or start
                )
            except (TypeError, ValueError):
                app.logger.warning(
                    "Skipping invalid segment: %r",
                    segment
                )
                continue

            if not text:
                continue

            # Prefer word timestamps that actually belong to this segment.
            # A segment's midpoint can be inaccurate for long segments, so use
            # overlap with its original interval and tolerate small boundary drift.
            if transcription_words:
                owned_words = []
                original_start, original_end = start, end
                for word in transcription_words:
                    try:
                        word_start = float(get_field(word, "start", -1))
                        word_end = float(get_field(word, "end", -1))
                    except (TypeError, ValueError):
                        continue
                    if word_start < 0 or word_end <= word_start:
                        continue
                    midpoint = (word_start + word_end) / 2.0
                    if original_start - 0.15 <= midpoint <= original_end + 0.15:
                        owned_words.append((word_start, word_end))

                if owned_words:
                    start = min(item[0] for item in owned_words)
                    end = max(item[1] for item in owned_words)

            # No global fixed delay: each subtitle follows its own detected speech.
            total_delay = SUBTITLE_DELAY_SECONDS + user_delay
            start = max(0.0, start + total_delay)
            end = max(start + 0.15, end + total_delay)

            raw_segments.append({
                "start": start,
                "end": end,
                "text": text
            })

        if not raw_segments:
            return jsonify(
                error=(
                    "စကားပြောစာသား မတွေ့ပါ။ "
                    "အသံပါသော တရုတ်ဗီဒီယိုနဲ့ ပြန်စမ်းပါ။"
                )
            ), 422

        # Prevent timestamp overlaps between adjacent subtitle entries. This only
        # trims a preceding cue at the next cue's speech onset; it does not apply
        # a global offset or extend subtitles through silence.
        raw_segments.sort(key=lambda item: (item["start"], item["end"]))
        for index in range(len(raw_segments) - 1):
            current = raw_segments[index]
            following = raw_segments[index + 1]
            if current["end"] > following["start"]:
                current["end"] = max(
                    current["start"] + 0.15,
                    following["start"]
                )

        # Translate with Gemini.
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        translated_lines = []
        batch_size = GEMINI_BATCH_SIZE

        for start_index in range(0, len(raw_segments), batch_size):
            batch = raw_segments[
                start_index:start_index + batch_size
            ]

            try:
                translations = translate_batch(model, batch)
                translated_lines.extend(translations)

            except Exception:
                app.logger.exception(
                    "Gemini translation failed at batch %s",
                    start_index
                )

                # Do not silently show untranslated Chinese as Burmese.
                return jsonify(
                    error=(
                        "မြန်မာဘာသာပြန်ခြင်း မအောင်မြင်ပါ။ "
                        "Render Logs မှာ Gemini error ကို စစ်ပါ။ "
                        "Quota ပြည့်နေပါက ခဏစောင့်ပြီး ပြန်စမ်းပါ။ "
                        "GEMINI_MODEL နဲ့ API Key ကိုလည်း စစ်ပါ။"
                    )
                ), 502

        # Prepare subtitle entries.
        items = []

        for index, segment in enumerate(raw_segments):
            translated_text = translated_lines[index]

            items.append({
                "index": index + 1,
                "time": (
                    f"{make_timestamp(segment['start'])} --> "
                    f"{make_timestamp(segment['end'])}"
                ),
                "text": format_subtitle(translated_text),
                "source": segment["text"]
            })

        return jsonify(
            items=items,
            notice=(
                "မြန်မာဘာသာပြန်ပြီးပါပြီ။ "
                "အချိန်များသည် မူရင်းစကားပြော segment များအပေါ် "
                "အခြေခံထားပါသည်။ စာတန်းများကို ပြန်စစ်ပြီး SRT ထုတ်ပါ။ "
                "ဗီဒီယိုထဲက မြင်ရသော တရုတ်စာတန်း OCR မပါသေးပါ။"
            )
        )

    except subprocess.TimeoutExpired:
        app.logger.exception("FFmpeg timed out")
        return jsonify(
            error="ဗီဒီယိုအသံထုတ်ယူချိန် ကျော်လွန်သွားပါတယ်။"
        ), 504

    except Exception:
        app.logger.exception("Video processing failed")
        return jsonify(
            error=(
                "လုပ်ဆောင်မှု မအောင်မြင်ပါ။ "
                "Render Logs၊ API Key နဲ့ API quota ကို စစ်ပါ။"
            )
        ), 500

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/api/export")
def export_srt():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])

    if not isinstance(items, list):
        return jsonify(error="Subtitle data မမှန်ပါ။"), 400

    output = []

    for item in items:
        if not isinstance(item, dict):
            continue

        timestamp = str(item.get("time", "")).strip()
        text = str(item.get("text", "")).strip()

        if "-->" not in timestamp or not text:
            continue

        output.append(
            f"{len(output) + 1}\n"
            f"{timestamp}\n"
            f"{text}"
        )

    srt_content = "\n\n".join(output)

    if srt_content:
        srt_content += "\n"

    return send_file(
        BytesIO(srt_content.encode("utf-8-sig")),
        mimetype="application/x-subrip",
        as_attachment=True,
        download_name="myanmar_subtitles.srt"
    )


@app.errorhandler(413)
def file_too_large(error):
    return jsonify(
        error="ဖိုင်အရွယ်အစား အရမ်းကြီးနေပါတယ်။"
    ), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
