# Qwen Tiny Voice Cloner

A lightweight Flask web interface for [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) — run multi-speaker, multi-language text-to-speech locally on a single consumer GPU.

Supports 9 built-in speakers across 11 languages with optional emotion/style control via natural language instructions. Includes sentence-level streaming so audio starts playing before the full text is synthesized.

## Requirements

- Python 3.12+
- NVIDIA GPU with 6 GB+ VRAM (tested on RTX 4070 Super 12 GB)
- CUDA 12.x

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U qwen-tts flask soundfile

# Optional — faster attention + lower VRAM (needs CUDA toolkit headers):
# pip install flash-attn --no-build-isolation
```

Model weights (~3 GB) download automatically on first run via Hugging Face Hub.

## Usage

```bash
source venv/bin/activate
python app.py
```

Open **http://localhost:5000** in your browser.

### API

**POST /api/synthesize** — full generation, returns a WAV file.

```bash
curl -X POST http://localhost:5000/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world!","speaker":"ryan","language":"english"}' \
  -o output.wav
```

**POST /api/stream** — sentence-level streaming, returns newline-delimited JSON with base64-encoded PCM chunks.

```bash
curl -X POST http://localhost:5000/api/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"First sentence. Second sentence.","speaker":"ryan","language":"english"}'
```

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *(required)* | Text to synthesize |
| `speaker` | string | `ryan` | Speaker voice (see below) |
| `language` | string | `auto` | Target language or `auto` |
| `instruct` | string | `null` | Emotion/style instruction, e.g. `"Speak cheerfully"` |

### Speakers

| Name | Description |
|------|-------------|
| `vivian` | Bright young female (Chinese) |
| `serena` | Warm, gentle young female (Chinese) |
| `uncle_fu` | Low, mellow male (Chinese) |
| `dylan` | Clear, natural male (Beijing Chinese) |
| `eric` | Lively, slightly husky male (Sichuan Chinese) |
| `ryan` | Dynamic male, strong rhythm (English) |
| `aiden` | Sunny American male (English) |
| `ono_anna` | Playful, light female (Japanese) |
| `sohee` | Warm, emotional female (Korean) |

### Languages

`auto`, `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`

## Credits

Based on the **Qwen3-TTS** family by the Qwen team at Alibaba:

- Model: [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- Repository: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- Paper: [arXiv:2601.15621](https://arxiv.org/abs/2601.15621)
- License: Apache 2.0

## License

Apache 2.0
