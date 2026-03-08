
import os
import json
import time
import re
import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from llama_cpp import Llama

# =========================
# Runtime / GPU settings
# =========================

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GEN_BACKEND = os.environ.get("GEN_BACKEND", "llama").strip().lower()
MODEL_PATH = os.environ.get("MODEL_PATH", "models/starling-lm-7b-alpha.Q4_K_M.gguf")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "35"))

CTX_LIMIT = 8192


# =========================
# Blog Settings
# =========================

@dataclass
class BlogSettings:
    topic: str
    audience: str = "general readers"
    tone: str = "clear, human, conversational"
    perspective: str = "second person"
    target_words: int = 2000
    sections: int = 8
    keywords: Optional[str] = None
    outfile: str = "blog_output.md"
    creativity: float = 0.9
    debug_chunks: bool = True


# =========================
# Model loader
# =========================

MODEL = None


def ensure_model():
    global MODEL
    if GEN_BACKEND != "llama":
        return
    if MODEL:
        return

    print("Loading GGUF model:", MODEL_PATH)

    MODEL = Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_LIMIT,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False
    )

    print("Model ready.")


# =========================
# Generation helper
# =========================

def generate(prompt, max_tokens=500, temperature=0.8, top_p=0.95):

    if GEN_BACKEND != "llama":
        return "[MOCK OUTPUT] " + prompt[:200]

    ensure_model()

    out = MODEL.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    text = out["choices"][0]["text"]
    return text.strip()


# =========================
# Blog pipeline
# =========================

def generate_title(topic):

    prompt = f"""
Write 5 compelling blog titles about:

{topic}

Titles should be:
- human sounding
- curiosity driven
- not clickbait
"""

    text = generate(prompt, 120)

    titles = [t.strip("-• ").strip() for t in text.splitlines() if len(t.strip()) > 5]

    if titles:
        return titles[0]

    return topic


def generate_outline(topic, sections):

    prompt = f"""
Create a blog outline with {sections} sections.

Topic:
{topic}

Return only section titles.
"""

    text = generate(prompt, 200)

    lines = [l.strip("-• ").strip() for l in text.splitlines() if len(l.strip()) > 4]

    return lines[:sections]


def generate_section(topic, section, words):

    prompt = f"""
Write a blog section.

Topic: {topic}
Section: {section}

Requirements:
- human sounding
- natural flow
- about {words} words
"""

    return generate(prompt, max_tokens=600)


def generate_conclusion(topic):

    prompt = f"""
Write a short thoughtful conclusion for a blog about:

{topic}
"""

    return generate(prompt, 200)


# =========================
# Blog Builder
# =========================

def run_blog(settings: BlogSettings):

    title = generate_title(settings.topic)
    outline = generate_outline(settings.topic, settings.sections)

    article = []
    article.append(f"# {title}")
    article.append("")

    words_per_section = max(120, settings.target_words // max(1, settings.sections))

    for sec in outline:

        text = generate_section(settings.topic, sec, words_per_section)

        article.append(f"## {sec}")
        article.append("")
        article.append(text)
        article.append("")

        if settings.debug_chunks:
            print("Generated section:", sec)

    article.append("## Conclusion")
    article.append("")
    article.append(generate_conclusion(settings.topic))

    final = "\n".join(article)

    with open(settings.outfile, "w", encoding="utf-8") as f:
        f.write(final)

    print("Saved:", settings.outfile)

    return final


# =========================
# Colab entry
# =========================

def main_colab(config_override=None):

    if config_override:
        cfg = config_override
    else:
        cfg = {
            "umbrella_topic": "example topic",
            "audience_hint": "general readers",
            "tone_hint": "conversational",
            "perspective_hint": "second person",
            "words": 2000,
            "sections": 8,
            "keywords_hint": "",
            "outfile": "blog_output.md",
            "creativity": 0.9
        }

    settings = BlogSettings(
        topic=cfg["umbrella_topic"],
        audience=cfg["audience_hint"],
        tone=cfg["tone_hint"],
        perspective=cfg["perspective_hint"],
        target_words=cfg["words"],
        sections=cfg["sections"],
        keywords=cfg.get("keywords_hint"),
        outfile=cfg["outfile"],
        creativity=cfg["creativity"]
    )

    return run_blog(settings)


# =========================
# CLI entry
# =========================

def main_cli():

    parser = argparse.ArgumentParser()

    parser.add_argument("--topic", required=True)
    parser.add_argument("--words", type=int, default=2000)
    parser.add_argument("--sections", type=int, default=8)
    parser.add_argument("--outfile", default="blog_output.md")

    args = parser.parse_args()

    settings = BlogSettings(
        topic=args.topic,
        target_words=args.words,
        sections=args.sections,
        outfile=args.outfile
    )

    run_blog(settings)


if __name__ == "__main__":

    if os.environ.get("USE_COLAB_INPUTS", "1") == "1":
        main_colab()
    else:
        main_cli()
