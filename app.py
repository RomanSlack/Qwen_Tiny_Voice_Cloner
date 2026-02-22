import io
import json
import re
import base64
import time

import numpy as np
import torch
import soundfile as sf
from flask import Flask, request, Response, send_file, jsonify
from qwen_tts import Qwen3TTSModel

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CUDA performance tuning
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Model loading (synchronous – avoids CUDA threading issues)
# ---------------------------------------------------------------------------
# Pick attention implementation: flash_attention_2 if available, else sdpa
try:
    import flash_attn  # noqa: F401
    _attn_impl = "flash_attention_2"
except ImportError:
    _attn_impl = "sdpa"

print(f"[*] Loading Qwen3-TTS model (attn={_attn_impl}) ...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation=_attn_impl,
)
SPEAKERS = model.get_supported_speakers()
LANGUAGES = model.get_supported_languages()
print(f"[+] Model loaded!  Speakers: {SPEAKERS}")

# Warmup pass – first inference is always slow (CUDA kernel caching)
print("[*] Warmup inference ...")
_t0 = time.time()
with torch.inference_mode():
    model.generate_custom_voice(text="Warmup.", speaker="ryan", language="english")
print(f"[+] Warmup done in {time.time() - _t0:.1f}s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(
    r'(?<=[.!?。！？；\n])\s*'
)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/status")
def status():
    return jsonify(status="ready", speakers=SPEAKERS, languages=LANGUAGES)


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Text is required"), 400

    speaker = data.get("speaker", "ryan")
    language = data.get("language", "auto")
    instruct = (data.get("instruct") or "").strip() or None

    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
        )

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav", download_name="output.wav")


@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Text is required"), 400

    speaker = data.get("speaker", "ryan")
    language = data.get("language", "auto")
    instruct = (data.get("instruct") or "").strip() or None

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    def generate():
        for i, sentence in enumerate(sentences):
            try:
                with torch.inference_mode():
                    wavs, sr = model.generate_custom_voice(
                        text=sentence,
                        speaker=speaker,
                        language=language,
                        instruct=instruct,
                    )
                pcm = wavs[0]  # float32/64 array in [-1, 1]
                pcm16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                audio_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")

                yield json.dumps({
                    "type": "audio",
                    "audio": audio_b64,
                    "sr": sr,
                    "index": i,
                    "total": len(sentences),
                    "text": sentence,
                }) + "\n"
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "error": str(e),
                    "index": i,
                }) + "\n"

        yield json.dumps({"type": "done"}) + "\n"

    return Response(
        generate(),
        mimetype="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ---------------------------------------------------------------------------
# Inline HTML / CSS / JS
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3 TTS</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c0c;--surface:#161616;--surface2:#1e1e1e;
  --border:#2a2a2a;--text:#e0e0e0;--muted:#777;
  --accent:#3b82f6;--accent-hover:#2563eb;
  --green:#22c55e;--red:#ef4444;--orange:#f59e0b;
  --radius:10px;
}
html{font-size:15px}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);
  min-height:100vh;display:flex;justify-content:center;
  padding:2rem 1rem;
}
.container{width:100%;max-width:640px}
h1{font-size:1.5rem;font-weight:600;margin-bottom:.25rem}
.subtitle{color:var(--muted);font-size:.85rem;margin-bottom:1.5rem}

/* Card */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.5rem;margin-bottom:1rem;
}

/* Form elements */
label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:.35rem;font-weight:500}
textarea,input[type=text],select{
  width:100%;padding:.6rem .75rem;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:.9rem;
  outline:none;transition:border-color .15s;
  font-family:inherit;
}
textarea:focus,input:focus,select:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:100px;line-height:1.5}
select{cursor:pointer;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%23777' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right .75rem center;
  padding-right:2rem;
}
select option{background:var(--surface2);color:var(--text)}

.row{display:flex;gap:.75rem;margin-top:.75rem}
.row>*{flex:1}

.instruct-row{margin-top:.75rem}

/* Buttons */
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:.5rem;
  padding:.65rem 1.5rem;border:none;border-radius:6px;
  font-size:.9rem;font-weight:500;cursor:pointer;
  transition:background .15s,opacity .15s;
  font-family:inherit;
}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover{background:var(--accent-hover)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
.btn-stop{background:var(--red);color:#fff;padding:.65rem 1rem}
.btn-stop:hover{background:#dc2626}

.actions{display:flex;gap:.75rem;margin-top:1rem;align-items:center}
.mode-toggle{
  display:flex;align-items:center;gap:.5rem;
  margin-left:auto;font-size:.8rem;color:var(--muted);
  cursor:pointer;user-select:none;
}
.mode-toggle input{accent-color:var(--accent);cursor:pointer;width:14px;height:14px}

/* Status bar */
.status-bar{
  display:flex;align-items:center;gap:.5rem;
  padding:.6rem .85rem;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);font-size:.8rem;color:var(--muted);
  margin-bottom:1rem;
}
.status-dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;
}
.status-dot.loading{background:var(--orange);animation:pulse 1.5s infinite}
.status-dot.ready{background:var(--green)}
.status-dot.error{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Audio output */
.output-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:1.25rem;margin-bottom:1rem;
  display:none;
}
.output-card.visible{display:block}

.output-label{font-size:.75rem;color:var(--muted);margin-bottom:.5rem;font-weight:500;text-transform:uppercase;letter-spacing:.05em}

audio{width:100%;height:40px;margin-top:.25rem;border-radius:6px}
audio::-webkit-media-controls-panel{background:var(--surface2)}

/* Progress */
.progress-wrap{margin-top:.75rem;display:none}
.progress-wrap.visible{display:block}
.progress-bar-bg{
  width:100%;height:4px;background:var(--surface2);border-radius:2px;overflow:hidden;
}
.progress-bar{height:100%;background:var(--accent);width:0%;transition:width .3s}
.progress-text{font-size:.75rem;color:var(--muted);margin-top:.35rem}

/* Generating indicator */
.generating{display:none;align-items:center;gap:.5rem;font-size:.85rem;color:var(--muted);margin-top:.75rem}
.generating.visible{display:flex}
.spinner{
  width:16px;height:16px;border:2px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;
  animation:spin .6s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* Timing */
.timing{font-size:.75rem;color:var(--muted);margin-top:.5rem}
</style>
</head>
<body>
<div class="container">
  <h1>Qwen3 TTS</h1>
  <p class="subtitle">Custom Voice &middot; Text-to-Speech</p>

  <!-- Status -->
  <div class="status-bar" id="statusBar">
    <div class="status-dot ready" id="statusDot"></div>
    <span id="statusText">Model ready</span>
  </div>

  <!-- Input -->
  <div class="card">
    <label for="textInput">Text</label>
    <textarea id="textInput" placeholder="Type or paste text to synthesize..."></textarea>

    <div class="row">
      <div>
        <label for="speakerSelect">Speaker</label>
        <select id="speakerSelect" disabled><option>Loading...</option></select>
      </div>
      <div>
        <label for="languageSelect">Language</label>
        <select id="languageSelect" disabled><option>Loading...</option></select>
      </div>
    </div>

    <div class="instruct-row">
      <label for="instructInput">Instruction <span style="color:var(--muted);font-weight:400">(optional &mdash; emotion / style)</span></label>
      <input type="text" id="instructInput" placeholder='e.g. "Speak cheerfully" or "Very sad tone"'>
    </div>

    <div class="actions">
      <button class="btn btn-primary" id="generateBtn" disabled>Generate</button>
      <button class="btn btn-stop" id="stopBtn" style="display:none">Stop</button>
      <label class="mode-toggle">
        <input type="checkbox" id="streamToggle" checked>
        Stream
      </label>
    </div>

    <div class="generating" id="generating">
      <div class="spinner"></div>
      <span id="generatingText">Generating...</span>
    </div>
  </div>

  <!-- Output -->
  <div class="output-card" id="outputCard">
    <div class="output-label">Output</div>
    <audio id="audioPlayer" controls></audio>
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar-bg"><div class="progress-bar" id="progressBar"></div></div>
      <div class="progress-text" id="progressText"></div>
    </div>
    <div class="timing" id="timing"></div>
  </div>
</div>

<script>
// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------
const $ = (s) => document.querySelector(s);
let abortController = null;
let audioCtx = null;
let scheduledSources = [];
let nextPlayTime = 0;

// -------------------------------------------------------------------------
// Model status polling
// -------------------------------------------------------------------------
async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    populateSelects(data.speakers, data.languages);
    $("#generateBtn").disabled = false;
  } catch {
    setTimeout(pollStatus, 2000);
  }
}

function populateSelects(speakers, languages) {
  const ss = $("#speakerSelect");
  const ls = $("#languageSelect");
  ss.innerHTML = "";
  ls.innerHTML = "";
  speakers.forEach((s) => {
    const o = document.createElement("option");
    o.value = s; o.textContent = s;
    if (s === "Ryan") o.selected = true;
    ss.appendChild(o);
  });
  ls.innerHTML = '<option value="Auto">Auto</option>';
  languages.forEach((l) => {
    const o = document.createElement("option");
    o.value = l; o.textContent = l;
    ls.appendChild(o);
  });
  ss.disabled = false;
  ls.disabled = false;
}

pollStatus();

// -------------------------------------------------------------------------
// Generate
// -------------------------------------------------------------------------
$("#generateBtn").addEventListener("click", generate);
$("#textInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate();
});

async function generate() {
  const text = $("#textInput").value.trim();
  if (!text) return;

  const streaming = $("#streamToggle").checked;
  const payload = {
    text,
    speaker: $("#speakerSelect").value,
    language: $("#languageSelect").value,
    instruct: $("#instructInput").value.trim() || null,
  };

  // UI: generating state
  setGenerating(true);
  const t0 = performance.now();

  if (streaming) {
    await generateStream(payload, t0);
  } else {
    await generateFull(payload, t0);
  }

  setGenerating(false);
}

// ---- Full (non-streaming) generation ------------------------------------
async function generateFull(payload, t0) {
  abortController = new AbortController();
  try {
    const res = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.error || "Generation failed");
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const player = $("#audioPlayer");
    player.src = url;
    player.play();
    showOutput();
    $("#progressWrap").classList.remove("visible");
    $("#timing").textContent = `Generated in ${((performance.now() - t0) / 1000).toFixed(1)}s`;
  } catch (e) {
    if (e.name !== "AbortError") alert("Error: " + e.message);
  }
}

// ---- Streaming generation -----------------------------------------------
async function generateStream(payload, t0) {
  abortController = new AbortController();

  // Reset Web Audio
  stopAudioPlayback();
  audioCtx = new AudioContext();
  nextPlayTime = 0;
  scheduledSources = [];

  // Collect all PCM chunks so we can build a final WAV for the <audio> player
  const allChunks = [];
  let sampleRate = 24000;

  showOutput();
  $("#progressWrap").classList.add("visible");
  $("#audioPlayer").removeAttribute("src");
  $("#audioPlayer").load();

  try {
    const res = await fetch("/api/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: abortController.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.error || "Streaming failed");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Process complete lines
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);

        if (msg.type === "audio") {
          sampleRate = msg.sr;

          // Decode base64 → Int16 → Float32
          const raw = atob(msg.audio);
          const int16 = new Int16Array(new Uint8Array([...raw].map(c => c.charCodeAt(0))).buffer);
          const float32 = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

          allChunks.push(float32);

          // Schedule playback via Web Audio API
          const buf = audioCtx.createBuffer(1, float32.length, sampleRate);
          buf.getChannelData(0).set(float32);
          const src = audioCtx.createBufferSource();
          src.buffer = buf;
          src.connect(audioCtx.destination);
          const startAt = Math.max(nextPlayTime, audioCtx.currentTime + 0.02);
          src.start(startAt);
          nextPlayTime = startAt + buf.duration;
          scheduledSources.push(src);

          // Progress
          const pct = Math.round(((msg.index + 1) / msg.total) * 100);
          $("#progressBar").style.width = pct + "%";
          $("#progressText").textContent = `Chunk ${msg.index + 1} / ${msg.total}: "${msg.text}"`;
          $("#generatingText").textContent = `Generating chunk ${msg.index + 1}/${msg.total}...`;
        }

        if (msg.type === "error") {
          console.error("Chunk error:", msg.error);
        }

        if (msg.type === "done") {
          $("#progressBar").style.width = "100%";
          $("#progressText").textContent = "Done";
        }
      }
    }

    // Build a combined WAV and put it in the <audio> element for replay / download
    if (allChunks.length) {
      const totalLen = allChunks.reduce((a, c) => a + c.length, 0);
      const combined = new Float32Array(totalLen);
      let offset = 0;
      for (const c of allChunks) { combined.set(c, offset); offset += c.length; }

      const wavBlob = float32ToWavBlob(combined, sampleRate);
      $("#audioPlayer").src = URL.createObjectURL(wavBlob);
    }

    $("#timing").textContent = `Streamed in ${((performance.now() - t0) / 1000).toFixed(1)}s`;

  } catch (e) {
    if (e.name !== "AbortError") alert("Error: " + e.message);
  }
}

// ---- Audio helpers -------------------------------------------------------
function stopAudioPlayback() {
  if (audioCtx) {
    scheduledSources.forEach(s => { try { s.stop(); } catch {} });
    audioCtx.close().catch(() => {});
  }
  scheduledSources = [];
  nextPlayTime = 0;
  $("#audioPlayer").pause();
}

function float32ToWavBlob(samples, sr) {
  const numCh = 1, bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numCh * bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  function writeStr(off, str) { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numCh, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s * 0x7FFF, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ---- UI helpers ----------------------------------------------------------
function setGenerating(on) {
  $("#generateBtn").disabled = on;
  $("#generating").classList.toggle("visible", on);
  $("#stopBtn").style.display = on ? "" : "none";
  if (!on) {
    $("#generatingText").textContent = "Generating...";
  }
}

function showOutput() {
  $("#outputCard").classList.add("visible");
}

// Stop button
$("#stopBtn").addEventListener("click", () => {
  if (abortController) abortController.abort();
  stopAudioPlayback();
  setGenerating(false);
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
