# Fashion Attribute Extraction Pipeline (BLIP + Ollama Mistral)

**Integrated, reliable pipeline to extract garment attributes from product images.**

This repository connects a local BLIP image-captioning model with an LLM (Mistral) served through **Ollama** to extract structured fashion attributes (neckline, silhouette, waistline, sleeves) from images listed in an Excel sheet. It includes robust image validation, retries, model readiness checks, and multiple output formats (CSV / JSON / Excel).

---

## Highlights

- Local captioning with **BLIP** (`transformers`) for deterministic image descriptions.
- Local LLM extraction using **Mistral via Ollama** (configurable model tag) for attribute parsing — with retries, auto-pull, and graceful fallbacks to regex rules.
- Input: Excel file containing image URLs (configurable path/sheet/column).
- Outputs: `results.csv`, `results.json`, `results_with_attrs.xlsx`, `results_skipped.json`.
- Optional plots of attribute distributions.
- Clear logging and robust network handling (timeouts, retries, backoff).

---

## Quickstart (TL;DR)

1. Ensure Ollama is installed and running locally.
2. Place your Excel file (default path set in the script) with a column of image URLs.
3. Install Python dependencies (see below).
4. Run the pipeline:

```bash
python local_main.py
```

---

## Table of Contents

- [Requirements](#requirements)  
- [Configuration](#configuration)  
- [How it works](#how-it-works)  
- [Usage](#usage)  
- [Integrating other LLM providers (OpenAI, etc.)](#integrating-other-llm-providers-openai-etc)  
- [Tips & Troubleshooting](#tips--troubleshooting)  
- [Output schema and example](#output-schema-and-example)  
- [License & Contributing](#license--contributing)  

---

## Requirements

The project targets Python 3.10+ and uses `transformers`, `torch`, and common data libraries. Install core deps via:

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: PyTorch must match your CUDA / CPU environment. Example install commands are present in `requirements.txt`. Choose the torch wheel matching your machine (GPU vs CPU).

Core Python packages (example):
- python-dotenv
- requests
- tqdm
- pandas
- matplotlib (optional, for plots)
- Pillow
- transformers
- torch (install the appropriate binary for your system)

---

## Files

- `local_main.py` — main pipeline (BLIP -> Ollama).  
- `requirements.txt` — core packages and instructions for torch.  
- `Best_Seller_Tags.xlsx` — (user-provided) input Excel.  
- Outputs generated in working directory: `results.csv`, `results.json`, `results_with_attrs.xlsx`, `results_skipped.json`.

---

## Configuration

You can configure the behavior either by editing constants at the top of `local_main.py` or by providing a `.env` file (the script will attempt to load `ENV_PATH` if present). Key configuration points:

- `EXCEL_PATH` — path to your input Excel file.  
- `SHEET_NAME` — sheet name containing image URLs.  
- `COL_NAME` — column name with image URLs.  
- `OLLAMA_API_URL` — default `http://localhost:11434/api/generate`.  
- `OLLAMA_MODEL_NAME` — Ollama model identifier (e.g. `mistral:7b-instruct-q4_K_M` or `mistral:latest`).  
- `AUTO_PULL_MODEL` — attempt to `ollama pull` model if it is not present locally.  
- `MAX_LINKS` — limit how many images to process (useful for testing).  
- `REQUEST_TIMEOUT`, `REQUEST_RETRIES`, `BACKOFF_FACTOR` — networking and retry tuning.  
- `CREATE_PLOTS` — whether to generate attribute distribution plots.  
- `LOCAL_BLIP_MODEL` — the transformers model ID (default: `Salesforce/blip-image-captioning-base`).

Example `.env` entries you may add:

```
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_MODEL_NAME=mistral:7b-instruct-q4_K_M
EXCEL_PATH=C:/Users/you/Desktop/Best_Seller_Tags.xlsx
```

---

## How it works (short)

1. Read URLs from the specified Excel sheet/column.  
2. Validate each URL (HEAD + small content sniff or image signature check). Skip invalid links.  
3. Download image bytes (with retries + backoff).  
4. Generate a caption using BLIP (FP16 on CUDA when available; graceful FP32 fallback).  
5. Send caption to Ollama/Mistral and ask for a JSON with four fields (neckline, silhouette, waistline, sleeves).  
   - If the LLM response cannot be parsed, the script falls back to a regex-based extractor that matches a curated keyword map.  
6. Save consolidated outputs to CSV, JSON and a new Excel with appended attributes, and log skipped URLs.

---

## Usage examples

Run the pipeline in the repository root (adjust `EXCEL_PATH` / `.env` when needed):

```bash
# Option 1: CPU-only or if torch already installed correctly
pip install -r requirements.txt
python local_main.py

# Option 2: Install a specific torch wheel (example CUDA 12.1)
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps
python local_main.py
```

If you prefer to only test a few images, set `MAX_LINKS` at the top of `local_main.py` to a small integer.

---

## Integrating other LLM providers (OpenAI, etc.)

If you prefer using a hosted provider (OpenAI, Anthropic, Cohere, etc.) for potentially better accuracy or larger models, you can replace the `generate_with_ollama()` implementation with a provider-specific call. The pipeline is intentionally modular in `generate_with_ollama()` / `extract_with_ollama()` so you can swap implementations. Suggestions:

- **OpenAI**: call the Chat Completions or Responses endpoint with the same `ATTR_PROMPT` payload and parse the returned text for JSON. Be sure to add robust retries, rate-limit handling, and max token limits. Use the model that fits your needs (e.g., gpt-4o / gpt-4 / gpt-3.5-turbo).  
- **Hybrid**: run a local Mistral (fast & private) for most images, and selectively escalate ambiguous captions to an external API for re-checking (costly but more accurate).  
- **Explainability**: store the raw LLM text output (already done by the script when verbose) so you can audit or re-run parsing logic later.

> **Privacy & Cost**: sending images or captions to hosted APIs can have privacy and cost implications—review your provider's data policies before sending sensitive product data.

---

## Example: How to swap to OpenAI (conceptual, not a drop-in)

```python
# inside generate_with_ollama() replace request with an HTTP call to OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role":"system","content":"You are a fashion expert."},
              {"role":"user","content":ATTR_PROMPT.format(caption=caption)}],
    temperature=0.0,
    max_tokens=256
)
text = resp.choices[0].message.content
```

Then reuse `_try_parse_json_from_text(text)` to convert text -> JSON. (You must install the official OpenAI Python SDK and set `OPENAI_API_KEY`.)

---

## Tips & Troubleshooting

- **Model fails to load / Ollama connection refused**: confirm Ollama is running (`ollama ps`), and your `OLLAMA_API_URL` is correct. The script supports auto-pull if `AUTO_PULL_MODEL=True` and Ollama's `/api/pull` endpoint is reachable.  
- **GPU OOM with BLIP**: try using `Salesforce/blip-image-captioning-base` (already used) or reduce `max_length` / `num_beams` in `BLIP_GENERATE_KWARGS`. Use the CPU fallback on smaller GPUs.  
- **Incorrect attribute extraction**: enable verbose mode and inspect the raw caption + LLM text output. The script falls back to a regex-based extractor when LLM output is unparseable. Consider adding more patterns to the keyword maps.  
- **Excel mapping not matching**: the script matches normalized URLs (trailing slash removed) when appending attributes back to the original sheet. If your sheet contains transformed / proxied URLs, match will fail — consider normalizing your input or adjust `normalize_url()`.
- **Rate limits & timeouts**: increase `OLLAMA_TIMEOUT`, `REQUEST_TIMEOUT`, or `REQUEST_RETRIES` for slow networks or heavy models.

---

## Output schema

Each output record contains:

```json
{
  "image_url": "https://.../image.jpg",
  "caption": "caption generated by BLIP (or fallback)",
  "neckline": "V-neck",
  "silhouette": "Sheath",
  "waistline": "Natural",
  "sleeves": "Sleeveless"
}
```

`Unknown` is used when a field cannot be reliably determined.

---

## Contributing & Extending

This repository is structured for easy extension:
- Add new attribute categories (length, pattern, fabric, color) by expanding the LLM prompt and regex maps.  
- Improve parsing by using an LLM response schema enforcement (tools like `pydantic` or strict JSON `format` instructions).  
- Add a cache layer for captions to avoid reprocessing identical images.

Pull requests, issues, and feature suggestions are welcome.

---

## License

This project is provided for demo / interview / prototype purposes. Use, modify and redistribute at your own discretion. Add an appropriate open source license if you plan to release publicly (e.g., MIT).

---

## Changelog (short)
- **v1.0** — Reliable BLIP + Ollama pipeline with image validation, retries, and Excel integration.
- **v1.1** — Add model auto-pull for Ollama, improved JSON extraction, balanced-brace parser fallback, and better logging.

---
