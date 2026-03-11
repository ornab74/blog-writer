
# Quantum-Entropic Long-Form Blog & Software Writer (Colab Notebook)

Notebook-based generator for detailed, human-sounding blog posts and software architecture documents.  
Runs fully in Google Colab with GPU acceleration.

Features pseudo-quantum style variation (via PennyLane), multi-agent consensus, micro-chunk section building, interstitial shadow nudges, optional peer-forum simulation, and self-revision passes.

## Overview

- Downloads and verifies Starling-LM-7B-alpha Q4_K_M (~4.1 GB)
- Installs llama-cpp-python with CUDA support (tested on Colab T4)
- Uses system entropy + small quantum circuit to influence phrasing, cadence, metaphor rate, etc.
- Multi-agent loop (Strategist → Researcher → Stylist → Editor → consensus synthesis)
- Sections built from 5–6 purpose-specific micro-chunks (hook, tension, example, mechanism, objection, takeaway)
- Dynamic micro-prompts and constraints injected between sections
- Optional: simulated personas debating sections + adversarial ghostwriter revisions
- Outputs markdown files + run manifest (settings + topics summary)

## Current Status (March 2026)

Model: Starling-LM-7B-alpha Q4_K_M  
Quantization: Q4_K_M  
Context: 4096 tokens  
Inference: llama-cpp-python + CUDA  
Speed: ~18–28 tok/s on T4 GPU  
Entropy: CPU/RAM/time + PennyLane default.qubit simulation  

Note: Starling-7B is from ~2023–2024. In 2026, stronger models (Qwen2.5-32B, Llama-3.3-70B, Gemma-3-27B, etc.) give noticeably better coherence and depth. Upgrade by changing MODEL_NAME / GGUF_URL in the download cell.

## How to Use in Colab

1. Open the notebook  
   (Link usually in the first cell as a badge)

2. Run cells sequentially:

   - Cell 1: Install pennylane + upgrade psutil
   - Cell 2: Install llama-cpp-python (CUDA wheel) + download Starling-7B Q4_K_M + verify SHA256
   - Cell 3 & 4: Configure topics, mode ("blog" / "software"), audience, tone, target words, sections
   - Cell 5: The full generator code — executes main_colab() automatically

3. Results appear in /content/:

   - blog_output.md or software_output.md
   - run_manifest.json (run summary)

## Configuration (edit before running generator)

```python
# Topic selection (from TOPIC_BANK or custom)
SELECTED_TOPICS = TOPIC_BANK["blog"][:1]           # or [:3] or custom list

# Mode & voice
RUN_MODE = "blog"                                  # "blog" | "software"
AUDIENCE_HINT = "Tech-curious builders and readers"
TONE_HINT = "clear, human, confident, and detailed"
PERSPECTIVE_HINT = "direct expert guidance with second-person phrasing when useful"

# Output control
TARGET_WORDS = 1800                                # 2200–3000 common for deep pieces
SECTION_COUNT = 6                                  # 7–9 for software write-ups
OUTFILE = "blog_output.md"                         # auto-changed for software mode
```

## Built-in Topic Bank

Blog mode
- How local-first apps change everyday workflows
- Why developer tooling wins or loses on onboarding
- What makes a technical tutorial actually readable
- The real tradeoffs between speed, quality, and maintainability
- How small teams ship polished software without huge process overhead
- Designing AI workflows that people actually trust

Software mode
- Build a production-ready FastAPI starter with auth, tests, and Docker
- Design a long-form AI-assisted documentation generator
- Create a modular plugin architecture for a Python desktop tool
- Implement a scalable job queue service with retries and observability
- Build a clean full-stack SaaS scaffold with billing and admin tools
- Create a local-first notes app with sync conflict handling

## Quick Model Upgrade Example (2026 recommendation)

In the download cell, replace:

```python
MODEL_NAME = "Qwen2.5-32B-Instruct-Q5_K_M.gguf"
GGUF_URL   = "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q5_K_M.gguf"
# Paste correct SHA256 from the model card
GGUF_SHA256 = "replace-with-actual-hash-from-hf"
CTX_SIZE = 8192   # or higher if model supports
```

Then re-run the setup cell.

## Philosophy

LLM long-form content often feels generic and repetitive.  
This notebook adds controlled variance through:

- real system metrics entropy
- quantum circuit simulation → style parameters
- agent disagreement + synthesis
- micro-chunk structure per section
- forward-looking shadow context nudges

The pattern still produces interesting results even when swapping in stronger base models.

## License
 GPL3
