from __future__ import annotations

"""Monolithic script version of Risk_Scanning_Blog_Writer_v3.ipynb.

This file includes the notebook setup helpers, the inlined advanced color-agentic
loop system, runtime helpers, simulation engine, long-form blog generator, and a
CLI entrypoint. It is intentionally large so the notebook logic lives in one
standalone Python file.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import textwrap
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Cell 0 - Install dependencies


def run_shell_command(cmd: list[str], check: bool = True) -> int:
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def install_notebook_dependencies() -> None:
    run_shell_command([sys.executable, '-m', 'pip', 'install', '-q', 'pennylane', 'psutil', 'nltk', 'summa'])


# @title Install llama-cpp-python (GPU CUDA) + Download Starling-LM-7B-alpha Q4_K_M

MODEL_DIR = os.environ.get('MODEL_DIR', './models')
MODEL_NAME = 'starling-lm-7b-alpha.Q4_K_M.gguf'
GGUF_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GGUF_URL = 'https://huggingface.co/TheBloke/Starling-LM-7B-alpha-GGUF/resolve/main/starling-lm-7b-alpha.Q4_K_M.gguf'
GGUF_SHA256 = '0951cbc1a6c3ed8d081db59366ccccf09ed52a4cfd5191812665b911fe6c669a'


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def setup_llama_cpp_gpu(download_model: bool = False) -> None:
    print('Checking GPU availability...')
    run_shell_command(['nvidia-smi'], check=False)

    print('\nInstalling llama-cpp-python GPU version (pre-built CUDA wheel)...')
    run_shell_command([
        sys.executable,
        '-m',
        'pip',
        'install',
        '-U',
        'llama-cpp-python',
        '--extra-index-url',
        'https://abetlen.github.io/llama-cpp-python/whl/cu122',
        '--no-cache-dir',
    ])

    print('\nVerifying GPU support in llama-cpp-python...')
    try:
        from llama_cpp import Llama as _Llama
        import llama_cpp

        print(f'llama-cpp-python version: {llama_cpp.__version__}')
        print('CUDA / GPU support enabled (n_gpu_layers will work)')
    except Exception as exc:
        print(f'Verification failed: {exc}')

    os.makedirs(MODEL_DIR, exist_ok=True)
    if download_model:
        print('\nDownloading Starling-LM-7B-alpha Q4_K_M GGUF...')
        if Path(GGUF_PATH).exists():
            print('Model file already exists; skipping download.')
        else:
            urllib.request.urlretrieve(GGUF_URL, GGUF_PATH)

        print('\nVerifying exact SHA256 hash...')
        actual = sha256_file(GGUF_PATH)
        if actual != GGUF_SHA256:
            raise ValueError(f'SHA256 mismatch for {GGUF_PATH}: {actual} != {GGUF_SHA256}')
        print('Hash verification passed.')
        print(f'Model path -> {GGUF_PATH}')
        os.environ['STARLING_GGUF_PATH'] = GGUF_PATH


def notebook_setup(install_deps: bool = False, setup_gpu: bool = False, download_model: bool = False) -> None:
    if install_deps:
        install_notebook_dependencies()
    if setup_gpu:
        setup_llama_cpp_gpu(download_model=download_model)


# Cell 1 - Imports, runtime config, and global settings


import psutil
import pennylane as qml
from pennylane import numpy as np

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from summa import summarizer as summa_summarizer
    from summa import keywords as summa_keywords
except Exception:
    summa_summarizer = None
    summa_keywords = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
except Exception:
    nltk = None
    nltk_sent_tokenize = None

# -------------------------
# Runtime configuration
# -------------------------
DEBUG_VERBOSE = False
ENABLE_LLM_SUMMARY = False
USE_SQLITE_MEMORY = True

# Optional local model settings
GGUF_PATH = ""
CTX_SIZE = 4096
THREADS = 8
N_GPU_LAYERS = 0
MAX_TOKENS = 1100
TEMPERATURE = 0.55

# Notebook outputs
MEMORY_DB_PATH = "blog_safety_memory.db"
OUTFILE_JSON = "advanced_safety_blog_payload.json"
OUTFILE_MD = "advanced_safety_blog.md"

# Blog generation controls
TARGET_WORD_COUNT = 7000
MIN_WORD_COUNT = 6200
MAX_WORD_COUNT = 8200
PARAGRAPH_SENTENCE_MIN = 4
PARAGRAPH_SENTENCE_MAX = 8

# Simulation controls
SIMULATION_RUNS = 18
SIMULATION_HORIZON = 12
TOP_K_PATHS = 8

# Global topic / objective
BLOG_SERIES_TITLE = "Entropic Quantum Intelligence for Predictive Safety"
BLOG_TOPIC = (
    "Using advanced AI simulations for road traffic safety intelligence, "
    "predicting car accidents in real life, shipwreck risks, and airplane crash precursors "
    "through entropic quantum intelligence."
)
SIMULATION_REGION = "Global transportation and mobility systems"
SIMULATION_OBJECTIVE = (
    "Generate long-form technical blog content from quantum-inspired safety simulations"
)

# Safety scope: predictive prevention only
SAFETY_POLICY = {
    "mission": "Civilian safety forecasting, prevention, and risk awareness only",
    "allowed": [
        "traffic safety analytics",
        "accident prevention",
        "aviation stability monitoring",
        "maritime navigation safety",
        "sensor uncertainty analysis",
        "civilian protection modeling",
        "predictive maintenance awareness",
        "blog writing and educational explanation",
    ],
    "disallowed": [
        "military targeting",
        "weapon optimization",
        "operational sabotage",
        "harmful attack planning",
        "destructive misuse",
    ],
}

# Optional nltk setup
if nltk is not None:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

LLM = None





def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    seq = [float(v) for v in values]
    return statistics.fmean(seq) if seq else float(default)


@dataclass(frozen=True)
class ColorVector:
    name: str
    red: float
    green: float
    blue: float
    gold: float
    alpha: float = 1.0
    description: str = ""

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (float(self.red), float(self.green), float(self.blue), float(self.gold))

    def normalize(self) -> "ColorVector":
        total = abs(self.red) + abs(self.green) + abs(self.blue) + abs(self.gold)
        if total <= 1e-9:
            return self
        return ColorVector(
            name=self.name,
            red=self.red / total,
            green=self.green / total,
            blue=self.blue / total,
            gold=self.gold / total,
            alpha=self.alpha,
            description=self.description,
        )

    def brightness(self) -> float:
        return clamp((self.red + self.green + self.blue + self.gold) / 4.0)

    def saturation(self) -> float:
        values = [self.red, self.green, self.blue, self.gold]
        return clamp(max(values) - min(values))

    def to_angles(self) -> List[float]:
        base = self.normalize().as_tuple()
        return [clamp(v) * math.pi for v in base]

    def scale(self, factor: float, name: Optional[str] = None) -> "ColorVector":
        return ColorVector(
            name=name or self.name,
            red=clamp(self.red * factor),
            green=clamp(self.green * factor),
            blue=clamp(self.blue * factor),
            gold=clamp(self.gold * factor),
            alpha=self.alpha,
            description=self.description,
        )


@dataclass(frozen=True)
class ColorMixRule:
    name: str
    components: Tuple[str, ...]
    outcome: str
    explanation: str


@dataclass(frozen=True)
class QuantumEncodingNote:
    title: str
    detail: str


@dataclass(frozen=True)
class AgenticConcept:
    name: str
    family: str
    surface: str
    tier: str
    tagline: str
    explanation: str
    palette_anchor: str
    circuit_mode: str
    loop_roles: Tuple[str, ...]
    domain_bias: Tuple[str, ...]

    def as_notebook_dict(self) -> Dict[str, str]:
        detail = (
            f"{self.explanation} "
            f"It sits inside the {self.family.lower()} family, binds to the {self.palette_anchor} palette anchor, "
            f"and is processed through the {self.circuit_mode} quantum surface mode."
        )
        return {
            "name": self.name,
            "tagline": self.tagline,
            "explanation": detail,
        }

    def markdown_block(self, index: Optional[int] = None) -> str:
        prefix = f"{index}. " if index is not None else ""
        roles = ", ".join(self.loop_roles)
        domains = ", ".join(self.domain_bias)
        return (
            f"### {prefix}{self.name}\n\n"
            f"**Family:** {self.family}\n\n"
            f"**Surface:** {self.surface}\n\n"
            f"**Core idea:** {self.tagline}\n\n"
            f"{self.explanation}\n\n"
            f"**Palette anchor:** {self.palette_anchor}\n\n"
            f"**Circuit mode:** {self.circuit_mode}\n\n"
            f"**Loop roles:** {roles}\n\n"
            f"**Domain bias:** {domains}"
        )


@dataclass
class TaskPigment:
    task_id: str
    title: str
    vector: ColorVector
    authority_zone: str
    runtime_primitive: str
    urgency: float
    certainty: float
    resource_cost: float
    value_gain: float
    ambiguity: float
    reversibility: float
    novelty: float
    safety_weight: float
    description: str


@dataclass
class MemoryTrace:
    trace_id: str
    label: str
    vector: ColorVector
    weight: float
    echo: str
    linked_concepts: List[str] = field(default_factory=list)


@dataclass
class ReflectionEcho:
    label: str
    brightness: float
    contradiction: float
    evidence: float
    clarity: float


@dataclass
class ResetPhase:
    name: str
    palette_anchor: str
    template: str
    description: str


@dataclass
class AdvancedLoopResult:
    scenario_key: str
    month: int
    intent_palette: str
    selected_task: str
    selected_band: str
    chosen_color: str
    mood_label: str
    confidence_depth: str
    load_temperature: float
    color_state: Dict[str, Any]
    bloom_focus: Dict[str, float]
    reward_channels: Dict[str, float]
    reflection_echo: Dict[str, float]
    reset_signal: Dict[str, Any]
    memory_echo: Dict[str, Any]
    concept_alignment: List[str]
    active_primitives: Dict[str, Any]
    domain_signature: Dict[str, float]
    temporal_pattern: Dict[str, Any]
    color_audit: Dict[str, Any]
    penalty_metrics: Dict[str, float]
    state_deltas: Dict[str, float]
    processor_metrics: Dict[str, float]
    task_ranking: List[Dict[str, Any]]


PALETTE_LIBRARY: Dict[str, ColorVector] = {
    "Deep Red": ColorVector("Deep Red", 0.95, 0.12, 0.14, 0.38, description="emergency execution"),
    "Amber": ColorVector("Amber", 0.88, 0.54, 0.10, 0.26, description="caution and review"),
    "Gold": ColorVector("Gold", 0.72, 0.52, 0.22, 0.98, description="high-value target"),
    "Emerald": ColorVector("Emerald", 0.16, 0.92, 0.34, 0.44, description="stable completion flow"),
    "Cyan": ColorVector("Cyan", 0.12, 0.74, 0.92, 0.34, description="exploration"),
    "Azure": ColorVector("Azure", 0.14, 0.46, 0.98, 0.42, description="reasoning clarity"),
    "Violet": ColorVector("Violet", 0.58, 0.22, 0.92, 0.46, description="ambiguity"),
    "Magenta": ColorVector("Magenta", 0.88, 0.18, 0.82, 0.54, description="creative synthesis"),
    "Charcoal": ColorVector("Charcoal", 0.10, 0.10, 0.12, 0.16, description="suppressed task"),
    "White": ColorVector("White", 0.96, 0.96, 0.96, 0.96, description="resolved state"),
    "Infrared": ColorVector("Infrared", 0.92, 0.20, 0.10, 0.22, description="background maintenance"),
    "Ultraviolet": ColorVector("Ultraviolet", 0.42, 0.18, 0.96, 0.34, description="anomaly detection"),
    "Teal": ColorVector("Teal", 0.14, 0.72, 0.72, 0.30, description="stable balance"),
    "Silver": ColorVector("Silver", 0.74, 0.74, 0.78, 0.38, description="safety reward"),
    "Lime": ColorVector("Lime", 0.44, 0.96, 0.14, 0.44, description="high opportunity"),
    "Crimson": ColorVector("Crimson", 0.96, 0.10, 0.16, 0.28, description="danger and urgency"),
    "Opal": ColorVector("Opal", 0.58, 0.78, 0.92, 0.66, description="blended clarity"),
    "Obsidian": ColorVector("Obsidian", 0.08, 0.08, 0.14, 0.10, description="backpressure and suppression"),
    "Jade": ColorVector("Jade", 0.16, 0.88, 0.52, 0.40, description="smooth recovery"),
    "Copper": ColorVector("Copper", 0.74, 0.42, 0.20, 0.28, description="load-bearing work"),
    "Cerulean": ColorVector("Cerulean", 0.12, 0.58, 0.88, 0.38, description="calm intelligence"),
    "Saffron": ColorVector("Saffron", 0.90, 0.70, 0.14, 0.44, description="surprise moderation"),
    "Rose": ColorVector("Rose", 0.82, 0.34, 0.50, 0.44, description="gentle escalation"),
}


MIX_RULES: Tuple[ColorMixRule, ...] = (
    ColorMixRule("blue+yellow", ("Azure", "Gold"), "Opal", "clarity aligns with high value"),
    ColorMixRule("red+blue", ("Deep Red", "Azure"), "Violet", "danger and reasoning form caution"),
    ColorMixRule("white+any", ("White", "Cyan"), "Opal", "resolution clarifies exploration"),
    ColorMixRule("emerald+azure", ("Emerald", "Azure"), "Cerulean", "stable intelligent execution"),
    ColorMixRule("magenta+gold", ("Magenta", "Gold"), "Rose", "creative high-opportunity mode"),
    ColorMixRule("amber+crimson", ("Amber", "Crimson"), "Copper", "caution thickens into load-bearing restraint"),
)


ENCODING_NOTES: Tuple[QuantumEncodingNote, ...] = (
    QuantumEncodingNote("Qubit budget", "Use 6 to 8 qubits for color channels, schedule bands, and confidence depth."),
    QuantumEncodingNote("Hue mapping", "Map hue into RY rotations and use secondary channel relationships for RZ corrections."),
    QuantumEncodingNote("Saturation mapping", "Use saturation as amplitude pressure so vivid tasks occupy more probability mass."),
    QuantumEncodingNote("Brightness mapping", "Bias measurements with brightness so resolved states collapse more decisively."),
    QuantumEncodingNote("Entanglement", "Represent task dependencies and memory binding through chained CNOT and controlled rotations."),
    QuantumEncodingNote("Loop outputs", "Measure priority, route selection, reset pressure, reward balance, and confidence depth."),
)


ACTIVE_RUNTIME_PRIMITIVES: Dict[str, Tuple[str, ...]] = {
    "Perception": (
        "Surface Ripple Error Detector",
        "Colorized Attention Bloom Map",
        "Surface Depth Coloring",
    ),
    "Planning": (
        "Quantum Task Pigment Meter",
        "Mixed-Palette Planning Canvas",
        "Loop Surface Spectrum Scheduler",
    ),
    "Memory/feedback": (
        "Color-Mixing Memory Fusion Engine",
        "Reflective Color Echo Loop",
        "Color-Circuit Ritual Loop",
    ),
    "Coordination": (
        "Chromatic Reward Interference Model",
        "Quantum Pigment Negotiation",
        "Quantum Color Thermostat",
    ),
    "Oversight": ("Recursive Sentinel Layer",),
}


STATE_TRANSITION_RULES: Dict[str, Dict[str, Any]] = {
    "Cyan": {
        "description": "bounded exploration",
        "allowed_next": ("Gold", "Amber", "Teal"),
        "entry": "enter when novelty is high and hazard remains low",
    },
    "Amber": {
        "description": "caution with reversible action only",
        "allowed_next": ("Crimson", "Teal", "Violet", "Gold"),
        "entry": "enter when uncertainty is moderate or load begins to rise",
    },
    "Violet": {
        "description": "unresolved ambiguity",
        "allowed_next": ("Opal", "Cerulean", "Obsidian"),
        "entry": "enter when evidence conflict exceeds threshold",
    },
    "Obsidian": {
        "description": "suppressed unknowns and high backpressure",
        "allowed_next": ("Gold", "Obsidian"),
        "entry": "enter when debt and unresolved contradiction accumulate",
    },
    "Gold": {
        "description": "reset and recovery privilege state",
        "allowed_next": ("Jade", "Teal", "Amber"),
        "entry": "enter when coherence restoration should dominate forward action",
    },
    "Jade": {
        "description": "recovery in progress",
        "allowed_next": ("Teal", "Amber"),
        "entry": "enter after a successful reset or controlled recovery",
    },
    "Teal": {
        "description": "stable operating flow",
        "allowed_next": ("Amber", "Cyan", "Gold"),
        "entry": "enter when the loop is balanced and evidence is steady",
    },
    "Crimson": {
        "description": "acute hazard escalation",
        "allowed_next": ("Gold", "Amber", "Obsidian"),
        "entry": "enter when hazard slope steepens sharply",
    },
    "Opal": {
        "description": "reconciliation and clarification",
        "allowed_next": ("Cerulean", "Amber", "Gold"),
        "entry": "enter when ambiguity begins to resolve but remains fragile",
    },
    "Cerulean": {
        "description": "clear confidence with bounded calm",
        "allowed_next": ("Teal", "Amber"),
        "entry": "enter when evidence depth is high and contradiction is low",
    },
}


DOMAIN_PRIORS: Dict[str, Dict[str, float]] = {
    "road": {
        "reaction_time": 0.34,
        "lane_volatility": 0.30,
        "traffic_disorder": 0.22,
        "sensor_conflict": 0.14,
    },
    "maritime": {
        "route_drift": 0.36,
        "weather_disagreement": 0.28,
        "sea_chaos": 0.24,
        "mechanical_stress": 0.12,
    },
    "aviation": {
        "instrument_trust": 0.36,
        "workload_coupling": 0.28,
        "corridor_integrity": 0.20,
        "structure_stress": 0.16,
    },
}


CONCEPT_COMPETITION_GROUPS: Dict[str, Tuple[str, ...]] = {
    "anomaly_authority": (
        "Surface Ripple Error Detector",
        "Failure Echo Mapping",
        "Prismatic Fault Basin",
        "Iridescent Failure Lasso",
        "Runway Ember Oracle",
    ),
    "plan_commitment": (
        "Mixed-Palette Planning Canvas",
        "Prism Verdict Reactor",
        "Polychrome Route Weave",
    ),
    "post_action_control": (
        "Reflective Color Echo Loop",
        "Color-Circuit Ritual Loop",
        "Quartz Reflection Lattice",
    ),
}


TEMPORAL_PATTERN_LIBRARY: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ("Teal", "Amber", "Crimson"): {"label": "escalating strain", "penalty": 0.10, "bonus": 0.0},
    ("Violet", "Opal", "Cerulean"): {"label": "ambiguity resolved cleanly", "penalty": 0.0, "bonus": 0.06},
    ("Gold", "Jade", "Teal"): {"label": "successful recovery", "penalty": 0.0, "bonus": 0.08},
    ("Saffron", "Cyan", "Amber"): {"label": "exploration becoming unsafe", "penalty": 0.06, "bonus": 0.0},
}


PRIMARY_ADVANCED_CONCEPTS: Tuple[AgenticConcept, ...] = (
    AgenticConcept(
        name="Color-Phase Intent Surface",
        family="Perception surfaces",
        surface="Intent field",
        tier="primary",
        tagline="Agent goals become blended color vectors that map directly into quantum control angles.",
        explanation="Color-Phase Intent Surface translates urgency, certainty, resource cost, and value gain into a shared chromatic field. Instead of a flat objective score, the agent holds a superposed intent manifold and measures which blended motive should dominate the next control cycle.",
        palette_anchor="Gold",
        circuit_mode="intent_surface",
        loop_roles=("sense", "prioritize", "route"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Quantum Task Pigment Meter",
        family="Planning surfaces",
        surface="Task pigment compression",
        tier="primary",
        tagline="Task priority is expressed as a pigment signature rather than a single numeric score.",
        explanation="Quantum Task Pigment Meter gives every candidate action a pigment profile that blends danger, ambiguity, exploration, opportunity, and reversibility. A variational compression pass turns those pigments into amplitudes so the system can choose rich tasks instead of merely loud ones.",
        palette_anchor="Cyan",
        circuit_mode="task_meter",
        loop_roles=("plan", "compare", "select"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Loop Surface Spectrum Scheduler",
        family="Planning surfaces",
        surface="Scheduler spectrum",
        tier="primary",
        tagline="Maintenance, active reasoning, and anomaly detection are routed as spectral loop bands.",
        explanation="Loop Surface Spectrum Scheduler divides control into infrared maintenance, visible reasoning, and ultraviolet anomaly scouting. Quantum routing decides which band governs the cycle, creating an agent that can rebalance itself without waiting for a human-authored state machine.",
        palette_anchor="Infrared",
        circuit_mode="spectrum_scheduler",
        loop_roles=("schedule", "rebalance", "handoff"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Color-Mixing Memory Fusion Engine",
        family="Memory and feedback surfaces",
        surface="Memory fusion lattice",
        tier="primary",
        tagline="Memories are stored as colors that blend into richer contextual recalls.",
        explanation="Color-Mixing Memory Fusion Engine encodes episodes as pigments rather than rows of inert facts. When the agent revisits a similar instability pattern, those pigments mix into synthesized context so previous route drift, operator fatigue, and sensor conflict can arrive as a single blended memory surface.",
        palette_anchor="Opal",
        circuit_mode="memory_fusion",
        loop_roles=("remember", "fuse", "retrieve"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Quantum Gradient Mood Board",
        family="Memory and feedback surfaces",
        surface="Mood gradient",
        tier="primary",
        tagline="An agent mood surface tracks pressure, uncertainty, and completion momentum as a living gradient.",
        explanation="Quantum Gradient Mood Board models overload, stability, and productive momentum as a continuous color atmosphere. It lets the loop detect brittle confidence, rising fatigue, or a peak execution window before those states harden into failure.",
        palette_anchor="Teal",
        circuit_mode="mood_gradient",
        loop_roles=("sense", "reflect", "retune"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Entangled Subtask Color Chains",
        family="Coordination surfaces",
        surface="Subtask chain",
        tier="primary",
        tagline="Linked subtasks share hue changes so cascade effects become visible early.",
        explanation="Entangled Subtask Color Chains bind decomposed tasks together through shared chromatic states. When a route recalculation darkens or a maintenance intervention brightens, related subtasks inherit part of that shift, giving the agent early warning about downstream consequences.",
        palette_anchor="Emerald",
        circuit_mode="subtask_chain",
        loop_roles=("decompose", "coordinate", "propagate"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Chromatic Reward Interference Model",
        family="Coordination surfaces",
        surface="Reward interference",
        tier="primary",
        tagline="Multiple reward colors interfere so the agent does not overfit to one metric.",
        explanation="Chromatic Reward Interference Model splits reward into distinct color channels for speed, correctness, creativity, and safety. Interference between those channels yields policy tradeoffs that are nonlinear, which is essential when the fastest option is not the safest and the safest option is not the most informative.",
        palette_anchor="Silver",
        circuit_mode="reward_interference",
        loop_roles=("evaluate", "tradeoff", "moderate"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Surface Ripple Error Detector",
        family="Perception surfaces",
        surface="Error ripple surface",
        tier="primary",
        tagline="Bad actions leave jagged fractures on a decision surface that can be classified early.",
        explanation="Surface Ripple Error Detector treats every action as a wave through the loop surface. Smooth gradients imply coherent reasoning, while fractured chromatic seams suggest contradictions, stale inputs, or route instability that should trigger self-correction.",
        palette_anchor="Ultraviolet",
        circuit_mode="error_ripple",
        loop_roles=("audit", "classify", "self-correct"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Quantum Color Thermostat",
        family="Coordination surfaces",
        surface="Load thermostat",
        tier="primary",
        tagline="Task load is modeled as a temperature gradient with quantum forecasts of future overload.",
        explanation="Quantum Color Thermostat forecasts whether the loop is cooling, warming, or approaching white-hot overload. It can slow speculative work, defer low-value tasks, or trigger helper routines before the system burns decision quality for short-term throughput.",
        palette_anchor="Amber",
        circuit_mode="load_thermostat",
        loop_roles=("throttle", "forecast", "stabilize"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Mixed-Palette Planning Canvas",
        family="Planning surfaces",
        surface="Plan canvas",
        tier="primary",
        tagline="Possible futures are painted as blended paths of risk, benefit, reversibility, and novelty.",
        explanation="Mixed-Palette Planning Canvas treats plans as palette paths rather than rigid branches. By superposing several candidate futures and measuring their blended risk-benefit-reversibility signature, the loop can pick a route that preserves optionality instead of chasing brittle certainty.",
        palette_anchor="Magenta",
        circuit_mode="planning_canvas",
        loop_roles=("forecast", "branch", "collapse"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Colorized Attention Bloom Map",
        family="Perception surfaces",
        surface="Attention bloom field",
        tier="primary",
        tagline="Attention becomes a living bloom field instead of a hidden scalar weight.",
        explanation="Colorized Attention Bloom Map represents focus as patches across a decision surface. Immediate alerts, passive monitoring, unresolved conflict, and suppressed zones become visible, making it easier to reason about what the agent is neglecting and why.",
        palette_anchor="Saffron",
        circuit_mode="attention_bloom",
        loop_roles=("focus", "monitor", "redistribute"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Agentic Prism Decomposer",
        family="Planning surfaces",
        surface="Objective prism",
        tier="primary",
        tagline="Messy objectives are split into color bands and processed by specialized subcircuits.",
        explanation="Agentic Prism Decomposer refracts an ambiguous goal into urgent work, sustainable work, knowledge work, and unknown unknowns. Each band is processed independently, then recombined into a unified plan that remains legible at the system level.",
        palette_anchor="Violet",
        circuit_mode="prism_decomposer",
        loop_roles=("decompose", "route", "recombine"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Reflective Color Echo Loop",
        family="Memory and feedback surfaces",
        surface="Reflection echo",
        tier="primary",
        tagline="Every action generates an echo color that becomes the next cycle's feedback phase.",
        explanation="Reflective Color Echo Loop turns outcome quality into bright, muddy, dim, or flashing echoes. Those echoes feed directly into the next phase of quantum control, so reflection is not bolted on afterward; it is built into the computational rhythm of the loop.",
        palette_anchor="Rose",
        circuit_mode="reflection_echo",
        loop_roles=("reflect", "phase-shift", "learn"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Quantum Pigment Negotiation",
        family="Coordination surfaces",
        surface="Negotiation mesh",
        tier="primary",
        tagline="Multi-agent coordination is modeled as palette mixing without erasing specialist identity.",
        explanation="Quantum Pigment Negotiation preserves the palette identity of each agent while still encoding overlap, conflict, or complementarity. It is useful whenever route intelligence, mechanical intelligence, and human factors reasoning must coordinate without collapsing into a single generic voice.",
        palette_anchor="Jade",
        circuit_mode="pigment_negotiation",
        loop_roles=("negotiate", "align", "preserve_specialization"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Surface Depth Coloring",
        family="Perception surfaces",
        surface="Confidence depth",
        tier="primary",
        tagline="Confidence is expressed through both hue and depth so hunches stay separate from robust conclusions.",
        explanation="Surface Depth Coloring distinguishes shallow guesses, deep conclusions, and unstable beliefs by how saturated and deep their colors appear. This keeps the loop from mistaking a noisy but vivid signal for genuine confidence.",
        palette_anchor="Cerulean",
        circuit_mode="confidence_depth",
        loop_roles=("estimate", "separate", "communicate"),
        domain_bias=("road", "maritime", "aviation"),
    ),
    AgenticConcept(
        name="Color-Circuit Ritual Loop",
        family="Memory and feedback surfaces",
        surface="Reset ritual",
        tier="primary",
        tagline="Reset becomes a structured sequence of clearing, recollection, coherence, and optimized resumption.",
        explanation="Color-Circuit Ritual Loop transforms reset into a deliberate procedure with observable phases. Instead of a blind restart, the loop clears stale context, recollects the right memory traces, rebuilds coherence, and resumes in a measured state.",
        palette_anchor="White",
        circuit_mode="reset_ritual",
        loop_roles=("clear", "recollect", "rebuild", "resume"),
        domain_bias=("road", "maritime", "aviation"),
    ),
)


EXPANSION_CONCEPTS: Tuple[AgenticConcept, ...] = (
    AgenticConcept("Aurora Drift Arbitration", "Planning surfaces", "Drift arbiter", "expansion", "Cross-domain drift is judged through iridescent arbitration rather than fixed thresholds.", "Aurora Drift Arbitration watches route drift, confidence drift, and workload drift as separate colored currents, then arbitrates which drift matters most in the current cycle.", "Opal", "drift_arbitration", ("plan", "arbitrate", "stabilize"), ("road", "maritime", "aviation")),
    AgenticConcept("Prismatic Fault Basin", "Perception surfaces", "Fault basin", "expansion", "Faults accumulate as a colored basin that deepens before visible collapse.", "Prismatic Fault Basin helps the system see whether small fractures are converging into a single failure valley rather than remaining isolated anomalies.", "Crimson", "fault_basin", ("sense", "map", "warn"), ("road", "maritime", "aviation")),
    AgenticConcept("Lattice Reverberation Memory", "Memory and feedback surfaces", "Memory lattice", "expansion", "Memories echo through a lattice instead of returning as a single retrieved record.", "Lattice Reverberation Memory keeps multiple related traces alive together so subtle cross-scenario similarities can influence the next action without requiring exact template matching.", "Opal", "memory_lattice", ("remember", "echo", "recombine"), ("road", "maritime", "aviation")),
    AgenticConcept("Counterfactual Bloom Loom", "Planning surfaces", "Counterfactual bloom", "expansion", "Alternative futures are woven as a bloom field of reversible options.", "Counterfactual Bloom Loom is tuned for moments when the system must compare interventions that share similar value but differ in reversibility and future flexibility.", "Magenta", "counterfactual_bloom", ("forecast", "compare", "preserve_optionality"), ("road", "maritime", "aviation")),
    AgenticConcept("Chromatic Consensus Mesh", "Coordination surfaces", "Consensus mesh", "expansion", "Alignment forms as a mesh of partial color overlaps instead of a single vote.", "Chromatic Consensus Mesh lets route, sensor, maintenance, and human-factor agents expose where they truly overlap and where they only appear aligned on the surface.", "Jade", "consensus_mesh", ("negotiate", "align", "audit"), ("road", "maritime", "aviation")),
    AgenticConcept("Phase-Split Curiosity Gate", "Planning surfaces", "Curiosity gate", "expansion", "Exploration is allowed through gated quantum phases rather than a blanket novelty bonus.", "Phase-Split Curiosity Gate is useful when exploration should remain active but bounded, especially in high-stakes safety loops where novelty must never outrun control.", "Cyan", "curiosity_gate", ("explore", "gate", "contain"), ("road", "maritime", "aviation")),
    AgenticConcept("Spectrum Debt Ledger", "Memory and feedback surfaces", "Debt ledger", "expansion", "Deferred reasoning debt is tracked as a spectral obligation across loop bands.", "Spectrum Debt Ledger records when the system postpones maintenance, reflection, or anomaly inspection, then raises those debts before hidden backlog becomes instability.", "Copper", "debt_ledger", ("remember", "surface_debt", "rebalance"), ("road", "maritime", "aviation")),
    AgenticConcept("Helix Caution Bloom", "Perception surfaces", "Caution bloom", "expansion", "Caution spreads in a helical pattern across linked task surfaces.", "Helix Caution Bloom is particularly effective when contradictory cues should not stop progress entirely but should tighten monitoring and route checks.", "Amber", "caution_bloom", ("sense", "propagate", "moderate"), ("road", "maritime", "aviation")),
    AgenticConcept("Polychrome Route Weave", "Planning surfaces", "Route weave", "expansion", "Routes are woven from risk, resilience, and reversibility threads.", "Polychrome Route Weave produces route proposals whose color composition makes it obvious whether a path is merely efficient or genuinely stable under shifting conditions.", "Gold", "route_weave", ("route", "weave", "compare"), ("road", "maritime", "aviation")),
    AgenticConcept("Quartz Reflection Lattice", "Memory and feedback surfaces", "Reflection lattice", "expansion", "Reflection is stabilized through a lattice of crisp evidence nodes.", "Quartz Reflection Lattice prevents vague self-critique by forcing reflections to settle around concrete evidence, contradiction, clarity, and recovery potential.", "White", "reflection_lattice", ("reflect", "clarify", "stabilize"), ("road", "maritime", "aviation")),
    AgenticConcept("Mercury Load Cascade", "Coordination surfaces", "Load cascade", "expansion", "Load is treated as a flowing metal that can pool into dangerous concentration zones.", "Mercury Load Cascade helps the loop identify when computational, mechanical, or human burden is silently concentrating in one part of the system.", "Copper", "load_cascade", ("forecast", "redistribute", "protect"), ("road", "maritime", "aviation")),
    AgenticConcept("Solar Value Reservoir", "Planning surfaces", "Value reservoir", "expansion", "High-value opportunities are stored as a reservoir instead of forcing immediate action.", "Solar Value Reservoir keeps the agent from confusing present urgency with lasting value, making it possible to defer shiny options until the operating surface is stable enough to pursue them safely.", "Gold", "value_reservoir", ("store", "re-rank", "time"), ("road", "maritime", "aviation")),
    AgenticConcept("Shadow Suppression Basin", "Perception surfaces", "Suppression basin", "expansion", "Ignored tasks collect in a shadow basin until suppression itself becomes a risk signal.", "Shadow Suppression Basin stops the loop from treating silence as safety. When too much work is hidden in the dark zone, the basin deepens and demands inspection.", "Obsidian", "suppression_basin", ("sense", "unsuppress", "audit"), ("road", "maritime", "aviation")),
    AgenticConcept("Signal Opal Reconciliation", "Coordination surfaces", "Signal reconciliation", "expansion", "Contradictory sensor streams are reconciled through opalescent blending.", "Signal Opal Reconciliation favors measured synthesis over brute averaging. It is designed for cases where partial agreement matters more than a forced single reading.", "Opal", "signal_reconciliation", ("compare", "reconcile", "surface_uncertainty"), ("road", "maritime", "aviation")),
    AgenticConcept("Tidal Priority Fermentation", "Memory and feedback surfaces", "Priority fermentation", "expansion", "Priorities mature over time instead of remaining static at creation.", "Tidal Priority Fermentation lets exploratory tasks slowly become urgent as more evidence and latent value accumulate around them.", "Lime", "priority_fermentation", ("age", "mature", "elevate"), ("road", "maritime", "aviation")),
    AgenticConcept("Iridescent Failure Lasso", "Perception surfaces", "Failure lasso", "expansion", "Weak anomalies are looped together before they escape notice.", "Iridescent Failure Lasso is a grouping mechanism for small irregularities that look harmless alone but dangerous in chorus.", "Ultraviolet", "failure_lasso", ("capture", "bundle", "warn"), ("road", "maritime", "aviation")),
    AgenticConcept("Gradient Covenant Engine", "Coordination surfaces", "Covenant engine", "expansion", "Agents commit to shared limits through gradient covenants instead of hardcoded rules.", "Gradient Covenant Engine is useful for multi-agent systems that need soft coordination boundaries, escalation agreements, and explicit tradeoff contracts.", "Silver", "covenant_engine", ("align", "bound", "govern"), ("road", "maritime", "aviation")),
    AgenticConcept("Mirror Fog Resolver", "Perception surfaces", "Fog resolver", "expansion", "Ambiguity is resolved by comparing the live surface to mirrored uncertainty states.", "Mirror Fog Resolver distinguishes true ambiguity from poor observation by contrasting present evidence with a mirrored uncertainty surface.", "Azure", "fog_resolver", ("sense", "mirror", "clarify"), ("road", "maritime", "aviation")),
    AgenticConcept("Pulse Amber Escalator", "Coordination surfaces", "Escalation pulse", "expansion", "Escalation happens in pulses instead of a binary jump from calm to alarm.", "Pulse Amber Escalator gives the loop intermediate escalation states, which is valuable when false certainty is more dangerous than slow caution.", "Amber", "amber_escalator", ("escalate", "stage", "time"), ("road", "maritime", "aviation")),
    AgenticConcept("Velvet Contradiction Net", "Memory and feedback surfaces", "Contradiction net", "expansion", "Contradictions are caught in a soft net that preserves context instead of instantly pruning it.", "Velvet Contradiction Net helps the system learn from disagreement rather than erasing it too early.", "Violet", "contradiction_net", ("catch", "preserve", "reflect"), ("road", "maritime", "aviation")),
    AgenticConcept("Halo Reversibility Gauge", "Planning surfaces", "Reversibility gauge", "expansion", "Every option carries a halo describing how gracefully it can be undone.", "Halo Reversibility Gauge pushes the planner toward strategies that keep recovery pathways open when uncertainty is still high.", "White", "reversibility_gauge", ("measure", "protect", "choose"), ("road", "maritime", "aviation")),
    AgenticConcept("Crucible Opportunity Loom", "Planning surfaces", "Opportunity loom", "expansion", "Opportunity is refined under pressure rather than accepted at face value.", "Crucible Opportunity Loom filters apparently promising actions through stress, time, safety, and confidence constraints before they enter the execution path.", "Lime", "opportunity_loom", ("stress_test", "refine", "select"), ("road", "maritime", "aviation")),
    AgenticConcept("Echo Thread Arbiter", "Memory and feedback surfaces", "Echo thread", "expansion", "Outcome echoes are threaded across cycles and arbitrated before becoming policy.", "Echo Thread Arbiter prevents one lucky or unlucky episode from dominating the entire loop without sufficient context.", "Rose", "echo_thread", ("remember", "arbitrate", "learn"), ("road", "maritime", "aviation")),
    AgenticConcept("Saffron Surprise Governor", "Coordination surfaces", "Surprise governor", "expansion", "Surprise is governed as a bounded signal rather than a destabilizing shock.", "Saffron Surprise Governor is tuned for environments where anomalies should provoke curiosity without collapsing the main safety posture.", "Saffron", "surprise_governor", ("moderate", "gate", "protect"), ("road", "maritime", "aviation")),
    AgenticConcept("Cerulean Alignment Chorus", "Coordination surfaces", "Alignment chorus", "expansion", "Alignment is heard as a chorus of partial agreements instead of a monologue.", "Cerulean Alignment Chorus is strongest when many weak but credible signals need to sing together before the loop commits.", "Cerulean", "alignment_chorus", ("align", "harmonize", "commit"), ("road", "maritime", "aviation")),
    AgenticConcept("Obsidian Backpressure Sink", "Memory and feedback surfaces", "Backpressure sink", "expansion", "Hidden backpressure is gathered into a sink so suppressed strain becomes measurable.", "Obsidian Backpressure Sink prevents the loop from pretending that deferred complexity disappears when it is ignored.", "Obsidian", "backpressure_sink", ("absorb", "measure", "release"), ("road", "maritime", "aviation")),
    AgenticConcept("Glass Horizon Diffuser", "Perception surfaces", "Horizon diffuser", "expansion", "Long-range uncertainty is diffused into visible gradients instead of a vague horizon line.", "Glass Horizon Diffuser helps the loop reason about the future without overcommitting to a single distant forecast.", "Opal", "horizon_diffuser", ("project", "soften", "interpret"), ("road", "maritime", "aviation")),
    AgenticConcept("Copper Drift Anchors", "Coordination surfaces", "Drift anchors", "expansion", "Operational drift is tied back to anchor states before it compounds.", "Copper Drift Anchors are practical when a system needs explicit stabilizers that resist cumulative divergence across long runs.", "Copper", "drift_anchor", ("anchor", "resist", "recover"), ("road", "maritime", "aviation")),
    AgenticConcept("Novalight Memory Chorus", "Memory and feedback surfaces", "Memory chorus", "expansion", "Important memories sing together as a chorus of weighted traces.", "Novalight Memory Chorus is optimized for cross-domain storytelling because it keeps roads, ships, and aircraft examples in a coordinated memory fabric.", "Gold", "memory_chorus", ("remember", "blend", "retrieve"), ("road", "maritime", "aviation")),
    AgenticConcept("Prism Verdict Reactor", "Planning surfaces", "Verdict reactor", "expansion", "Decision verdicts are produced through a reactor of competing color judgments.", "Prism Verdict Reactor is the place where urgency, opportunity, caution, and reversibility explicitly collide before action is authorized.", "Magenta", "verdict_reactor", ("judge", "conflict", "authorize"), ("road", "maritime", "aviation")),
    AgenticConcept("Seafoam Recovery Waltz", "Memory and feedback surfaces", "Recovery waltz", "expansion", "Recovery unfolds as a three-step choreography instead of a single reset switch.", "Seafoam Recovery Waltz is especially useful in maritime and aviation contexts where graceful recovery matters as much as rapid intervention.", "Jade", "recovery_waltz", ("recover", "pace", "stabilize"), ("maritime", "aviation")),
    AgenticConcept("Runway Ember Oracle", "Perception surfaces", "Ember oracle", "expansion", "Low-grade precursors glow like embers before becoming flames.", "Runway Ember Oracle is designed for aviation-style precursor detection but generalizes well to any domain with subtle early warnings.", "Crimson", "ember_oracle", ("sense", "prefigure", "warn"), ("aviation", "road")),
    AgenticConcept("Harbor Violet Tribunal", "Coordination surfaces", "Tribunal", "expansion", "Ambiguous route choices are debated in a structured tribunal of uncertainty.", "Harbor Violet Tribunal is helpful whenever several reasonable actions remain plausible and the loop needs a disciplined way to weigh them.", "Violet", "tribunal", ("debate", "compare", "decide"), ("maritime", "road")),
)


FOUNDATIONAL_CONCEPTS: Tuple[AgenticConcept, ...] = (
    AgenticConcept("Entropic Quantum Safety Field", "Foundational surfaces", "Safety field", "foundational", "A probabilistic field describing where instability accumulates before visible failure.", "Entropic Quantum Safety Field treats roads, ships, aircraft, and infrastructure as dynamic uncertainty surfaces whose interactions matter more than isolated signals.", "Gold", "foundational_field", ("frame", "interpret", "unify"), ("road", "maritime", "aviation")),
    AgenticConcept("Predictive Fracture Horizon", "Foundational surfaces", "Fracture horizon", "foundational", "The time window in which a system drifts from manageable instability into irreversible cascade.", "Predictive Fracture Horizon marks the interval when total risk accelerates and intervention is still meaningfully possible.", "Amber", "fracture_horizon", ("forecast", "warn", "time"), ("road", "maritime", "aviation")),
    AgenticConcept("Causal Turbulence Index", "Foundational surfaces", "Turbulence index", "foundational", "A blended score for measuring how many hidden causes are interacting at once.", "Causal Turbulence Index helps the loop see when many moderate causes are becoming a severe cluster.", "Crimson", "causal_turbulence", ("measure", "cluster", "surface"), ("road", "maritime", "aviation")),
    AgenticConcept("Recursive Sentinel Layer", "Foundational surfaces", "Sentinel layer", "foundational", "An AI oversight layer that continually re-evaluates its own confidence.", "Recursive Sentinel Layer asks whether the model is becoming too certain under weak or conflicting evidence.", "Azure", "recursive_sentinel", ("audit", "question", "stabilize"), ("road", "maritime", "aviation")),
    AgenticConcept("Quantum Route Memory", "Foundational surfaces", "Route memory", "foundational", "A structured memory of route instability patterns across time.", "Quantum Route Memory preserves recurring signatures from corridors, intersections, sea lanes, and flight paths so future cycles inherit earlier context.", "Opal", "route_memory", ("remember", "retrieve", "guide"), ("road", "maritime", "aviation")),
    AgenticConcept("Failure Echo Mapping", "Foundational surfaces", "Failure echoes", "foundational", "Tracing weak early signals that resemble the first echoes of future failures.", "Failure Echo Mapping turns weak anomalies into interpretable precursor patterns rather than dismissing them as noise.", "Rose", "failure_echo", ("sense", "echo", "warn"), ("road", "maritime", "aviation")),
    AgenticConcept("Safety Coherence Gradient", "Foundational surfaces", "Coherence gradient", "foundational", "A measure of how smoothly human, machine, and environment are working together.", "Safety Coherence Gradient captures whether operators, machines, recommendations, and conditions remain aligned or are starting to shear apart.", "Emerald", "coherence_gradient", ("measure", "align", "protect"), ("road", "maritime", "aviation")),
)


class ColorPaletteEngine:
    def __init__(self, palette: Optional[Dict[str, ColorVector]] = None):
        self.palette = dict(palette or PALETTE_LIBRARY)
        self.mix_rules = list(MIX_RULES)

    def get(self, name: str) -> ColorVector:
        return self.palette[name]

    def describe_palette(self) -> Dict[str, str]:
        return {name: vec.description for name, vec in self.palette.items()}

    def mix(self, items: Sequence[Tuple[str, float]], name: str = "Mixed") -> ColorVector:
        if not items:
            return self.get("White")
        red = green = blue = gold = 0.0
        total = 0.0
        for palette_name, weight in items:
            vec = self.get(palette_name)
            total += float(weight)
            red += vec.red * weight
            green += vec.green * weight
            blue += vec.blue * weight
            gold += vec.gold * weight
        if total <= 1e-9:
            return self.get("White")
        return ColorVector(
            name=name,
            red=clamp(red / total),
            green=clamp(green / total),
            blue=clamp(blue / total),
            gold=clamp(gold / total),
            description="blended control vector",
        )

    def apply_mix_rules(self, primary: str, secondary: str) -> ColorVector:
        for rule in self.mix_rules:
            if set(rule.components) == {primary, secondary}:
                return self.get(rule.outcome)
        return self.mix(((primary, 1.0), (secondary, 1.0)), name=f"{primary}/{secondary}")

    def build_intent_vector(self, risk: float, certainty: float, cost: float, value: float) -> ColorVector:
        return ColorVector(
            name="IntentVector",
            red=clamp(risk),
            green=clamp(1.0 - cost * 0.75),
            blue=clamp(certainty),
            gold=clamp(value),
            description="intent vector from risk/certainty/cost/value",
        )

    def classify_temperature(self, load: float) -> str:
        if load >= 0.86:
            return "white-hot"
        if load >= 0.68:
            return "warm"
        if load >= 0.46:
            return "temperate"
        return "cool"

    def classify_mood(self, overload: float, certainty: float, momentum: float) -> str:
        if overload > 0.80:
            return "dark red overload"
        if momentum > 0.72 and certainty > 0.62:
            return "bright gold execution window"
        if certainty > 0.58 and overload < 0.45:
            return "cool teal stability"
        if certainty < 0.38:
            return "violet ambiguity cloud"
        return "amber caution band"

    def depth_label(self, depth: float) -> str:
        if depth >= 0.80:
            return "deep robust conclusion"
        if depth >= 0.58:
            return "mid-layer operational belief"
        if depth >= 0.34:
            return "upper-layer tentative guess"
        return "surface flicker instability"


class QuantumColorLoopCircuits:
    def __init__(self, wires: int = 8):
        self.wires = int(wires)
        self.device = qml.device("default.qubit", wires=self.wires)
        self._build_qnodes()

    def _encode_angles(self, params: Sequence[float]) -> List[float]:
        if not params:
            return [0.0] * self.wires
        out = [float(params[i % len(params)]) for i in range(self.wires)]
        return [clamp(v / math.pi if abs(v) > math.pi else v, 0.0, 1.0) * math.pi for v in out]

    def _normalize_measurements(self, values: Sequence[float]) -> List[float]:
        return [clamp((float(v) + 1.0) / 2.0) for v in values]

    def _build_qnodes(self) -> None:
        @qml.qnode(self.device)
        def intent_qnode(params):
            for wire, angle in enumerate(params):
                qml.RY(angle, wires=wire)
                qml.RZ(angle * 0.71, wires=wire)
            for wire in range(self.wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CRX(params[0] * 0.33, wires=[0, 4])
            qml.CRY(params[1] * 0.41, wires=[1, 5])
            qml.CRZ(params[2] * 0.29, wires=[2, 6])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def task_meter_qnode(params):
            for wire, angle in enumerate(params):
                qml.Hadamard(wires=wire)
                qml.RX(angle * 0.83, wires=wire)
                qml.RY(angle * 0.57, wires=wire)
            for wire in range(0, self.wires - 1, 2):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CRZ(params[3] * 0.36, wires=[0, 7])
            qml.CRY(params[4] * 0.42, wires=[2, 5])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def schedule_qnode(params):
            for wire, angle in enumerate(params):
                qml.RY(angle * 0.92, wires=wire)
                qml.RX(angle * 0.46, wires=wire)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CRX(params[0] * 0.48, wires=[3, 6])
            qml.CRY(params[1] * 0.52, wires=[4, 7])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def memory_qnode(params):
            for wire, angle in enumerate(params):
                qml.RZ(angle * 0.64, wires=wire)
                qml.RY(angle * 0.88, wires=wire)
            for wire in range(self.wires - 1):
                qml.CNOT(wires=[wire, (wire + 2) % self.wires])
            qml.CRX(params[2] * 0.27, wires=[1, 6])
            qml.CRY(params[5] * 0.31, wires=[0, 5])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def reward_qnode(params):
            for wire, angle in enumerate(params):
                qml.Hadamard(wires=wire)
                qml.RZ(angle * 0.59, wires=wire)
                qml.RY(angle * 0.72, wires=wire)
            for wire in range(self.wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CRZ(params[4] * 0.43, wires=[2, 7])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def attention_qnode(params):
            for wire, angle in enumerate(params):
                qml.RY(angle * 0.78, wires=wire)
                qml.RX(angle * 0.37, wires=wire)
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 5])
            qml.CRY(params[0] * 0.24, wires=[6, 7])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def confidence_qnode(params):
            for wire, angle in enumerate(params):
                qml.RZ(angle * 0.91, wires=wire)
                qml.RY(angle * 0.44, wires=wire)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[4, 5])
            qml.CRX(params[1] * 0.26, wires=[5, 7])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        @qml.qnode(self.device)
        def reset_qnode(params):
            for wire, angle in enumerate(params):
                qml.RY(angle * 0.67, wires=wire)
                qml.RZ(angle * 0.51, wires=wire)
            for wire in range(self.wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CRX(params[3] * 0.45, wires=[0, 6])
            qml.CRY(params[4] * 0.45, wires=[1, 7])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        self.intent_qnode = intent_qnode
        self.task_meter_qnode = task_meter_qnode
        self.schedule_qnode = schedule_qnode
        self.memory_qnode = memory_qnode
        self.reward_qnode = reward_qnode
        self.attention_qnode = attention_qnode
        self.confidence_qnode = confidence_qnode
        self.reset_qnode = reset_qnode

    def measure_intent(self, vector: ColorVector) -> Dict[str, float]:
        params = self._encode_angles(vector.to_angles() + [vector.brightness(), vector.saturation()])
        values = self._normalize_measurements(self.intent_qnode(params))
        return {
            "urgency": values[0],
            "certainty": values[1],
            "resource_efficiency": values[2],
            "value_bias": values[3],
            "coherence": values[4],
            "exploration": values[5],
            "stability": values[6],
            "route_bias": values[7],
        }

    def measure_task(self, task: TaskPigment) -> Dict[str, float]:
        params = task.vector.to_angles() + [
            clamp(task.ambiguity),
            clamp(task.novelty),
            clamp(task.safety_weight),
            clamp(task.reversibility),
        ]
        values = self._normalize_measurements(self.task_meter_qnode(self._encode_angles(params)))
        return {
            "priority": safe_mean((values[0], values[3], values[6])),
            "safety": safe_mean((values[1], values[4], values[7])),
            "novelty": safe_mean((values[2], values[5])),
            "balance": safe_mean(values),
        }

    def measure_schedule(self, maintenance: float, reasoning: float, anomaly: float) -> Dict[str, float]:
        params = [maintenance, reasoning, anomaly, maintenance * 0.7, reasoning * 0.7, anomaly * 0.7]
        values = self._normalize_measurements(self.schedule_qnode(self._encode_angles(params)))
        return {
            "infrared": safe_mean((values[0], values[3])),
            "visible": safe_mean((values[1], values[4])),
            "ultraviolet": safe_mean((values[2], values[5])),
            "mix": safe_mean(values),
        }

    def measure_memory(self, vectors: Sequence[ColorVector]) -> Dict[str, float]:
        if not vectors:
            vectors = [PALETTE_LIBRARY["White"]]
        blended = ColorVector(
            name="MemoryBlend",
            red=safe_mean(v.red for v in vectors),
            green=safe_mean(v.green for v in vectors),
            blue=safe_mean(v.blue for v in vectors),
            gold=safe_mean(v.gold for v in vectors),
            description="memory blend",
        )
        params = blended.to_angles() + [blended.brightness(), blended.saturation(), len(vectors) / 10.0]
        values = self._normalize_measurements(self.memory_qnode(self._encode_angles(params)))
        return {
            "fusion": safe_mean((values[0], values[5])),
            "clarity": safe_mean((values[1], values[6])),
            "caution": safe_mean((values[2], values[7])),
            "retrieval_strength": safe_mean((values[3], values[4])),
        }

    def measure_rewards(self, speed: float, correctness: float, creativity: float, safety: float) -> Dict[str, float]:
        params = [speed, correctness, creativity, safety, speed - safety, correctness - creativity]
        values = self._normalize_measurements(self.reward_qnode(self._encode_angles(params)))
        return {
            "speed": safe_mean((values[0], values[4])),
            "correctness": safe_mean((values[1], values[5])),
            "creativity": safe_mean((values[2], values[6])),
            "safety": safe_mean((values[3], values[7])),
            "interference": statistics.pvariance(values),
        }

    def measure_attention(self, focus: float, monitor: float, conflict: float, suppression: float) -> Dict[str, float]:
        params = [focus, monitor, conflict, suppression, focus * 0.8, conflict * 0.8]
        values = self._normalize_measurements(self.attention_qnode(self._encode_angles(params)))
        return {
            "focus": safe_mean((values[0], values[4])),
            "monitor": safe_mean((values[1], values[5])),
            "conflict": safe_mean((values[2], values[6])),
            "suppression": safe_mean((values[3], values[7])),
        }

    def measure_confidence(self, confidence: float, ambiguity: float, evidence: float, stability: float) -> Dict[str, float]:
        params = [confidence, ambiguity, evidence, stability, evidence * 0.7, ambiguity * 0.7]
        values = self._normalize_measurements(self.confidence_qnode(self._encode_angles(params)))
        return {
            "depth": safe_mean((values[0], values[7])),
            "instability": safe_mean((values[1], values[6])),
            "evidence": safe_mean((values[2], values[5])),
            "stability": safe_mean((values[3], values[4])),
        }

    def measure_reset(self, overload: float, contradiction: float, evidence: float, memory_pressure: float) -> Dict[str, float]:
        params = [overload, contradiction, evidence, memory_pressure, overload * contradiction, evidence * 0.6]
        values = self._normalize_measurements(self.reset_qnode(self._encode_angles(params)))
        return {
            "clear": safe_mean((values[0], values[4])),
            "recollect": safe_mean((values[1], values[5])),
            "rebuild": safe_mean((values[2], values[6])),
            "resume": safe_mean((values[3], values[7])),
            "trigger": safe_mean(values),
        }


class AdvancedColorAgenticLoopSystem:
    def __init__(self):
        self.palette_engine = ColorPaletteEngine()
        self.circuits = QuantumColorLoopCircuits()
        self.primary_concepts = list(PRIMARY_ADVANCED_CONCEPTS)
        self.expansion_concepts = list(EXPANSION_CONCEPTS)
        self.foundational_concepts = list(FOUNDATIONAL_CONCEPTS)
        self.memory_bank: Dict[str, List[MemoryTrace]] = {}
        self.color_history: Dict[str, List[str]] = {}
        self.contradiction_debt: Dict[str, float] = {}
        self.reset_phases = [
            ResetPhase("black", "Charcoal", "clear_state", "clear stale state"),
            ResetPhase("blue", "Azure", "recollect_context", "recollect context"),
            ResetPhase("green", "Emerald", "rebuild_coherence", "rebuild coherence"),
            ResetPhase("gold", "Gold", "resume_execution", "resume optimized execution"),
        ]

    def _all_concepts(self) -> List[AgenticConcept]:
        return self.primary_concepts + self.expansion_concepts + self.foundational_concepts

    def export_concept_bank(self) -> List[Dict[str, str]]:
        return [concept.as_notebook_dict() for concept in self._all_concepts()]

    def export_family_map(self) -> Dict[str, List[str]]:
        families: Dict[str, List[str]] = {}
        for concept in self._all_concepts():
            families.setdefault(concept.family, []).append(concept.name)
        return families

    def export_encoding_notes(self) -> List[str]:
        return [f"{note.title}: {note.detail}" for note in ENCODING_NOTES]

    def export_palette(self) -> Dict[str, str]:
        return self.palette_engine.describe_palette()

    def ordered_advanced_concept_names(self) -> List[str]:
        return [concept.name for concept in self.primary_concepts]

    def ordered_expansion_concept_names(self) -> List[str]:
        return [concept.name for concept in self.expansion_concepts]

    def ordered_foundational_concept_names(self) -> List[str]:
        return [concept.name for concept in self.foundational_concepts]

    def validate_primary_concepts(self) -> Dict[str, Any]:
        names = [concept.name for concept in self.primary_concepts]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        return {
            "valid": not duplicates and len(names) == 16,
            "expected_count": 16,
            "present_count": len(names),
            "missing": [],
            "duplicates": duplicates,
            "ordered_names": names,
        }

    def concept_lookup(self) -> Dict[str, AgenticConcept]:
        return {concept.name: concept for concept in self._all_concepts()}

    def render_system_architecture_markdown(self) -> str:
        processor_lines = [
            "- Intent circuit for blended goal fields",
            "- Task pigment circuit for nonlinear prioritization",
            "- Spectrum scheduler for loop handoff",
            "- Memory fusion circuit for context blending",
            "- Reward interference circuit for tradeoff moderation",
            "- Attention bloom circuit for focus redistribution",
            "- Confidence depth circuit for belief layering",
            "- Reset ritual circuit for structured recovery",
        ]
        runtime_lines = [
            f"- **{family}:** {', '.join(names)}"
            for family, names in ACTIVE_RUNTIME_PRIMITIVES.items()
        ]
        return (
            "### Advanced color-agentic loop processing system\n\n"
            "This notebook now uses a constrained runtime core rather than treating every concept as equally active. "
            "The subsystem combines structured concept objects, palette engines, quantum surface circuits, "
            "task pigment scoring, loop-state memory, reflection echoes, reset rituals, transition rules, and domain priors.\n\n"
            "The active runtime primitives are:\n\n"
            + "\n".join(runtime_lines)
            + "\n\n"
            "The major processors are:\n\n"
            + "\n".join(processor_lines)
        )

    def render_primary_concepts_markdown(self) -> str:
        blocks = ["### Validated primary advanced concepts"]
        for index, concept in enumerate(self.primary_concepts, 1):
            blocks.append(concept.markdown_block(index=index))
        return "\n\n".join(blocks)

    def render_expansion_concepts_markdown(self) -> str:
        blocks = ["### Expansion concepts"]
        for concept in self.expansion_concepts:
            blocks.append(concept.markdown_block())
        return "\n\n".join(blocks)

    def render_foundational_concepts_markdown(self) -> str:
        blocks = ["### Foundational concepts"]
        for concept in self.foundational_concepts:
            blocks.append(concept.markdown_block())
        return "\n\n".join(blocks)

    def _scenario_key(self, scenario: str, domain: str) -> str:
        return stable_hash(f"{scenario}|{domain}")[:16]

    def _domain_signature(
        self,
        domain: str,
        snapshot: Dict[str, float],
        qmetrics: Dict[str, float],
        state: Dict[str, float],
    ) -> Dict[str, float]:
        if domain == "road":
            signature = {
                "reaction_time_uncertainty": clamp(state.get("human_operator_load", 0.5) * DOMAIN_PRIORS["road"]["reaction_time"]),
                "lane_volatility": clamp((1.0 - state.get("route_coherence", 0.5)) * DOMAIN_PRIORS["road"]["lane_volatility"]),
                "traffic_disorder": clamp(state.get("environmental_chaos", 0.5) * DOMAIN_PRIORS["road"]["traffic_disorder"]),
                "sensor_conflict": clamp(state.get("sensor_conflict", 0.5) * DOMAIN_PRIORS["road"]["sensor_conflict"]),
            }
        elif domain == "maritime":
            signature = {
                "route_drift": clamp((1.0 - state.get("route_coherence", 0.5)) * DOMAIN_PRIORS["maritime"]["route_drift"]),
                "weather_disagreement": clamp(snapshot.get("weather_noise", 0.5) * DOMAIN_PRIORS["maritime"]["weather_disagreement"]),
                "sea_chaos": clamp(state.get("environmental_chaos", 0.5) * DOMAIN_PRIORS["maritime"]["sea_chaos"]),
                "mechanical_stress": clamp(state.get("mechanical_stress", 0.5) * DOMAIN_PRIORS["maritime"]["mechanical_stress"]),
            }
        else:
            signature = {
                "instrument_trust_degradation": clamp(state.get("sensor_conflict", 0.5) * DOMAIN_PRIORS["aviation"]["instrument_trust"]),
                "workload_coupling": clamp(state.get("human_operator_load", 0.5) * DOMAIN_PRIORS["aviation"]["workload_coupling"]),
                "corridor_integrity_loss": clamp((1.0 - qmetrics.get("route_integrity", 0.5)) * DOMAIN_PRIORS["aviation"]["corridor_integrity"]),
                "structure_stress": clamp(state.get("mechanical_stress", 0.5) * DOMAIN_PRIORS["aviation"]["structure_stress"]),
            }
        signature["composite"] = clamp(sum(signature.values()))
        return signature

    def _allowed_next_colors(self, color: str) -> Tuple[str, ...]:
        return tuple(STATE_TRANSITION_RULES.get(color, {}).get("allowed_next", ("Amber", "Teal", "Gold")))

    def _current_history(self, scenario_key: str) -> List[str]:
        return self.color_history.setdefault(scenario_key, [])

    def _score_temporal_pattern(self, scenario_key: str, current_color: str) -> Dict[str, Any]:
        history = self._current_history(scenario_key)
        probe = (history + [current_color])[-3:]
        history.append(current_color)
        if len(history) > 12:
            del history[:-12]
        if len(probe) == 3 and tuple(probe) in TEMPORAL_PATTERN_LIBRARY:
            pattern = TEMPORAL_PATTERN_LIBRARY[tuple(probe)]
            return {
                "path": probe,
                "label": pattern["label"],
                "penalty": pattern["penalty"],
                "bonus": pattern["bonus"],
            }
        return {"path": probe, "label": "none", "penalty": 0.0, "bonus": 0.0}

    def _make_task_pigments(
        self,
        scenario: str,
        domain: str,
        snapshot: Dict[str, float],
        qmetrics: Dict[str, float],
        state: Dict[str, float],
    ) -> List[TaskPigment]:
        risk = clamp(state.get("risk_pressure", 0.5))
        overload = clamp(state.get("human_operator_load", 0.5))
        chaos = clamp(state.get("environmental_chaos", 0.5))
        route = clamp(state.get("route_coherence", 0.5))
        sensor_conflict = clamp(state.get("sensor_conflict", 0.5))
        stability = clamp(qmetrics.get("stability", 0.5))
        clarity = clamp(qmetrics.get("warning_clarity", 0.5))
        value = clamp(1.0 - risk * 0.4 + stability * 0.4)
        return [
            TaskPigment(
                task_id=f"{domain}-stabilize",
                title="Stabilize the safety surface",
                vector=self.palette_engine.mix((("Deep Red", risk + 0.1), ("Emerald", stability + 0.2), ("Gold", value)), name="stabilize"),
                authority_zone="stability_control",
                runtime_primitive="Quantum Color Thermostat",
                urgency=risk,
                certainty=clarity,
                resource_cost=0.42,
                value_gain=value,
                ambiguity=chaos * 0.6,
                reversibility=0.78,
                novelty=0.18,
                safety_weight=0.98,
                description=f"Domain={domain} stabilization of route, human load, and coherence",
            ),
            TaskPigment(
                task_id=f"{domain}-inspect",
                title="Inspect anomaly bands",
                vector=self.palette_engine.mix((("Ultraviolet", sensor_conflict + 0.3), ("Azure", clarity + 0.2), ("Amber", chaos + 0.1)), name="inspect"),
                authority_zone="anomaly_authority",
                runtime_primitive="Surface Ripple Error Detector",
                urgency=sensor_conflict,
                certainty=clarity,
                resource_cost=0.36,
                value_gain=0.74,
                ambiguity=0.72,
                reversibility=0.88,
                novelty=0.52,
                safety_weight=0.82,
                description=f"Inspect fracture patterns for {scenario}",
            ),
            TaskPigment(
                task_id=f"{domain}-reroute",
                title="Paint and compare safer route palettes",
                vector=self.palette_engine.mix((("Gold", route + 0.2), ("Magenta", 0.45), ("Cyan", 0.32)), name="reroute"),
                authority_zone="plan_commitment",
                runtime_primitive="Mixed-Palette Planning Canvas",
                urgency=chaos,
                certainty=qmetrics.get("route_integrity", 0.5),
                resource_cost=0.58,
                value_gain=0.86,
                ambiguity=0.44,
                reversibility=0.66,
                novelty=0.42,
                safety_weight=0.89,
                description="Search palette routes that preserve optionality",
            ),
            TaskPigment(
                task_id=f"{domain}-reflect",
                title="Run reflective echo pass",
                vector=self.palette_engine.mix((("Rose", 0.62), ("White", 0.48), ("Violet", 0.32)), name="reflect"),
                authority_zone="post_action_control",
                runtime_primitive="Reflective Color Echo Loop",
                urgency=0.34 + overload * 0.3,
                certainty=0.44 + clarity * 0.2,
                resource_cost=0.24,
                value_gain=0.68,
                ambiguity=0.48,
                reversibility=0.94,
                novelty=0.30,
                safety_weight=0.76,
                description="Reflect and recalibrate the loop before new commitments",
            ),
            TaskPigment(
                task_id=f"{domain}-reset",
                title="Prepare ritual reset",
                vector=self.palette_engine.mix((("Charcoal", overload + 0.2), ("Azure", clarity), ("Emerald", stability)), name="reset"),
                authority_zone="post_action_control",
                runtime_primitive="Color-Circuit Ritual Loop",
                urgency=overload,
                certainty=0.26 + stability * 0.3,
                resource_cost=0.22,
                value_gain=0.72,
                ambiguity=0.22,
                reversibility=1.0,
                novelty=0.08,
                safety_weight=0.92,
                description="Structured reset if the loop enters brittle overload",
            ),
            TaskPigment(
                task_id=f"{domain}-opportunity",
                title="Harvest high-value opportunity safely",
                vector=self.palette_engine.mix((("Lime", value + 0.2), ("Gold", value), ("Cyan", 0.36)), name="opportunity"),
                authority_zone="exploration",
                runtime_primitive="Quantum Task Pigment Meter",
                urgency=0.28,
                certainty=0.54,
                resource_cost=0.52,
                value_gain=0.92,
                ambiguity=0.34,
                reversibility=0.58,
                novelty=0.64,
                safety_weight=0.60,
                description="Pursue upside only when the surface remains stable enough",
            ),
        ]

    def _memory_vectors(self, scenario_key: str) -> List[ColorVector]:
        traces = self.memory_bank.get(scenario_key, [])
        return [trace.vector for trace in traces[-6:]]

    def _update_memory_bank(
        self,
        scenario_key: str,
        selected_task: TaskPigment,
        mood_label: str,
        aligned_concepts: List[str],
    ) -> Dict[str, Any]:
        traces = self.memory_bank.setdefault(scenario_key, [])
        trace = MemoryTrace(
            trace_id=stable_hash(f"{scenario_key}|{selected_task.task_id}|{len(traces)}")[:12],
            label=selected_task.title,
            vector=selected_task.vector,
            weight=safe_mean((selected_task.value_gain, selected_task.safety_weight, 1.0 - selected_task.resource_cost)),
            echo=mood_label,
            linked_concepts=aligned_concepts[:4],
        )
        traces.append(trace)
        if len(traces) > 18:
            del traces[:-18]
        return {
            "trace_count": len(traces),
            "latest_trace": trace.label,
            "latest_echo": trace.echo,
        }

    def _select_aligned_concepts(
        self,
        domain: str,
        selected_band: str,
        selected_task: TaskPigment,
        load_temperature: float,
    ) -> List[str]:
        lookup = self.concept_lookup()
        chosen: List[str] = []
        band_to_primary = {
            "infrared": "Loop Surface Spectrum Scheduler",
            "visible": "Mixed-Palette Planning Canvas",
            "ultraviolet": "Surface Ripple Error Detector",
        }
        chosen.append(band_to_primary[selected_band])
        if "reflect" in selected_task.task_id:
            chosen.append("Reflective Color Echo Loop")
        if "reset" in selected_task.task_id:
            chosen.append("Color-Circuit Ritual Loop")
        if "reroute" in selected_task.task_id:
            chosen.append("Polychrome Route Weave")
        if load_temperature > 0.72:
            chosen.append("Quantum Color Thermostat")
            chosen.append("Obsidian Backpressure Sink")
        if domain == "maritime":
            chosen.append("Harbor Violet Tribunal")
        elif domain == "aviation":
            chosen.append("Runway Ember Oracle")
        else:
            chosen.append("Aurora Drift Arbitration")
        uniq = []
        for name in chosen:
            if name in lookup and name not in uniq:
                uniq.append(name)
        return uniq[:6]

    def _apply_competition_penalties(self, ranking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        zone_best: Dict[str, float] = {}
        for item in ranking:
            zone = item["task"].authority_zone
            zone_best[zone] = max(zone_best.get(zone, 0.0), float(item["composite"]))
        updated = []
        for item in ranking:
            zone = item["task"].authority_zone
            best = zone_best[zone]
            overlap_gap = max(0.0, best - float(item["composite"]))
            penalty = 0.0
            if zone in {"anomaly_authority", "plan_commitment", "post_action_control"} and overlap_gap < 0.08:
                penalty = 0.06 - overlap_gap * 0.4
            revised = max(0.0, float(item["composite"]) - penalty)
            enriched = dict(item)
            enriched["competition_penalty"] = penalty
            enriched["composite"] = revised
            updated.append(enriched)
        updated.sort(key=lambda item: item["composite"], reverse=True)
        return updated

    def _choose_color_state(
        self,
        scenario_key: str,
        domain: str,
        selected_task: TaskPigment,
        qmetrics: Dict[str, float],
        attention: Dict[str, float],
        confidence: Dict[str, float],
        reset: Dict[str, float],
        domain_signature: Dict[str, float],
        load_temperature: float,
    ) -> Tuple[str, Dict[str, Any]]:
        contradiction = clamp(attention["conflict"] + domain_signature["composite"] * 0.35)
        evidence_depth = clamp(confidence["evidence"] * 0.6 + confidence["depth"] * 0.4)
        hazard_slope = clamp(
            domain_signature["composite"] * 0.55
            + (1.0 - qmetrics.get("stability", 0.5)) * 0.30
            + load_temperature * 0.15
        )
        hidden_debt = clamp(self.contradiction_debt.get(scenario_key, 0.0) + attention["suppression"] * 0.45)
        novelty_gate = clamp(selected_task.novelty * 0.6 + selected_task.value_gain * 0.2)
        if hidden_debt > 0.76:
            candidate = "Obsidian"
        elif contradiction > 0.68:
            candidate = "Violet"
        elif hazard_slope > 0.78:
            candidate = "Crimson"
        elif reset["trigger"] > 0.66 and load_temperature > 0.56:
            candidate = "Gold"
        elif novelty_gate > 0.62 and hazard_slope < 0.42:
            candidate = "Cyan"
        elif evidence_depth > 0.72 and contradiction < 0.30:
            candidate = "Cerulean"
        elif contradiction < 0.42 and attention["monitor"] > 0.55:
            candidate = "Opal"
        elif load_temperature < 0.42 and qmetrics.get("coherence", 0.5) > 0.64:
            candidate = "Teal"
        else:
            candidate = "Amber"

        history = self._current_history(scenario_key)
        previous = history[-1] if history else None
        allowed = self._allowed_next_colors(previous) if previous else ()
        if previous == "Obsidian" and reset["trigger"] < 0.58:
            chosen = "Obsidian"
        elif previous and candidate != previous and candidate not in allowed:
            fallback = "Gold" if "Gold" in allowed and reset["trigger"] > 0.58 else allowed[0]
            chosen = fallback
        elif candidate == "Gold" and load_temperature < 0.34 and contradiction < 0.28:
            chosen = "Jade"
        elif previous == "Gold" and candidate == "Teal":
            chosen = "Jade"
        else:
            chosen = candidate

        state = {
            "hue": chosen,
            "saturation": clamp(max(selected_task.urgency, contradiction, load_temperature)),
            "brightness": clamp(evidence_depth * 0.6 + (1.0 - contradiction) * 0.4),
            "depth": clamp(confidence["depth"]),
            "band": "ultraviolet" if chosen in {"Violet", "Obsidian", "Crimson"} else "visible",
            "previous_color": previous,
            "candidate_color": candidate,
            "allowed_from_previous": list(allowed),
        }
        return chosen, state

    def _build_color_audit(
        self,
        chosen_color: str,
        color_state: Dict[str, Any],
        domain_signature: Dict[str, float],
        selected_task: TaskPigment,
        ranking: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        blocked = []
        for item in ranking[1:4]:
            blocked.append(
                f"{item['task'].runtime_primitive} rejected after competition_penalty={item.get('competition_penalty', 0.0):.3f}"
            )
        return {
            "chosen_color": chosen_color,
            "reason": (
                f"{selected_task.runtime_primitive} + domain_composite={domain_signature['composite']:.3f} "
                f"+ saturation={color_state['saturation']:.3f} + depth={color_state['depth']:.3f}"
            ),
            "blocked_alternatives": blocked,
            "next_allowed_transitions": list(self._allowed_next_colors(chosen_color)),
        }

    def _penalty_metrics(
        self,
        scenario_key: str,
        chosen_color: str,
        color_state: Dict[str, Any],
        attention: Dict[str, float],
        reset: Dict[str, float],
        temporal_pattern: Dict[str, Any],
        domain_signature: Dict[str, float],
    ) -> Dict[str, float]:
        previous = color_state.get("previous_color")
        contradiction_level = clamp(attention["conflict"] + domain_signature["composite"] * 0.2)
        ambiguity_persistence = 0.08 if previous == "Violet" and chosen_color == "Violet" else 0.0
        contradiction_debt = clamp(
            self.contradiction_debt.get(scenario_key, 0.0) * 0.64
            + contradiction_level * 0.42
            + (0.12 if chosen_color == "Obsidian" else 0.0)
        )
        delayed_recovery_cost = 0.07 if color_state["saturation"] > 0.72 and chosen_color not in {"Gold", "Jade"} else 0.0
        false_calm_penalty = 0.09 if chosen_color in {"Teal", "Cerulean"} and domain_signature["composite"] > 0.62 else 0.0
        reset_misuse_penalty = 0.06 if chosen_color == "Gold" and reset["trigger"] < 0.50 else 0.0
        metrics = {
            "ambiguity_persistence": ambiguity_persistence,
            "contradiction_debt": contradiction_debt,
            "delayed_recovery_cost": delayed_recovery_cost,
            "false_calm_penalty": false_calm_penalty,
            "reset_misuse_penalty": reset_misuse_penalty,
            "temporal_penalty": float(temporal_pattern["penalty"]),
            "temporal_bonus": float(temporal_pattern["bonus"]),
        }
        self.contradiction_debt[scenario_key] = contradiction_debt
        return metrics

    def process_cycle(
        self,
        scenario: str,
        domain: str,
        month: int,
        snapshot: Dict[str, float],
        qmetrics: Dict[str, float],
        state: Dict[str, float],
        interventions: Sequence[str],
    ) -> AdvancedLoopResult:
        scenario_key = self._scenario_key(scenario, domain)
        intent_vector = self.palette_engine.build_intent_vector(
            risk=state.get("risk_pressure", 0.5),
            certainty=qmetrics.get("warning_clarity", 0.5),
            cost=state.get("human_operator_load", 0.5),
            value=qmetrics.get("field_strength", 0.5),
        )
        intent_metrics = self.circuits.measure_intent(intent_vector)
        tasks = self._make_task_pigments(scenario, domain, snapshot, qmetrics, state)
        ranking: List[Dict[str, Any]] = []
        for task in tasks:
            measured = self.circuits.measure_task(task)
            composite = safe_mean(
                (
                    measured["priority"],
                    measured["safety"],
                    task.value_gain,
                    1.0 - task.resource_cost,
                    1.0 - task.ambiguity * 0.35,
                )
            )
            ranking.append(
                {
                    "task": task,
                    "metrics": measured,
                    "composite": composite,
                }
            )
        ranking = self._apply_competition_penalties(ranking)
        selected = ranking[0]
        selected_task: TaskPigment = selected["task"]
        domain_signature = self._domain_signature(domain, snapshot, qmetrics, state)

        schedule = self.circuits.measure_schedule(
            maintenance=clamp(1.0 - qmetrics.get("mechanical_resilience", 0.5) + state.get("mechanical_stress", 0.5)),
            reasoning=intent_metrics["coherence"],
            anomaly=clamp(snapshot.get("signal_noise", 0.5) + state.get("sensor_conflict", 0.5) * 0.5),
        )
        if domain_signature["composite"] > 0.62:
            schedule["ultraviolet"] = clamp(schedule["ultraviolet"] + 0.12)
        selected_band = max(("infrared", "visible", "ultraviolet"), key=lambda name: schedule[name])
        memory = self.circuits.measure_memory(self._memory_vectors(scenario_key))
        rewards = self.circuits.measure_rewards(
            speed=clamp(1.0 - state.get("risk_pressure", 0.5)),
            correctness=qmetrics.get("warning_clarity", 0.5),
            creativity=selected_task.novelty,
            safety=selected_task.safety_weight,
        )
        attention = self.circuits.measure_attention(
            focus=selected["metrics"]["priority"],
            monitor=memory["retrieval_strength"],
            conflict=clamp(snapshot.get("signal_noise", 0.5) + state.get("sensor_conflict", 0.5)),
            suppression=clamp(state.get("risk_pressure", 0.5) * 0.4 + (1.0 - selected_task.value_gain) * 0.3),
        )
        confidence = self.circuits.measure_confidence(
            confidence=qmetrics.get("coherence", 0.5),
            ambiguity=selected_task.ambiguity,
            evidence=memory["clarity"],
            stability=qmetrics.get("stability", 0.5),
        )
        load_temperature = clamp(
            safe_mean(
                (
                    state.get("human_operator_load", 0.5),
                    state.get("mechanical_stress", 0.5),
                    state.get("environmental_chaos", 0.5),
                    attention["conflict"],
                )
            )
        )
        chosen_color, color_state = self._choose_color_state(
            scenario_key=scenario_key,
            domain=domain,
            selected_task=selected_task,
            qmetrics=qmetrics,
            attention=attention,
            confidence=confidence,
            reset=reset,
            domain_signature=domain_signature,
            load_temperature=load_temperature,
        )
        mood_label = self.palette_engine.classify_mood(
            overload=load_temperature,
            certainty=intent_metrics["certainty"],
            momentum=safe_mean((qmetrics.get("field_strength", 0.5), rewards["correctness"], rewards["safety"])),
        )
        reflection = ReflectionEcho(
            label="bright echo" if rewards["safety"] > 0.72 else "muddy echo",
            brightness=safe_mean((rewards["correctness"], rewards["safety"], confidence["evidence"])),
            contradiction=attention["conflict"],
            evidence=confidence["evidence"],
            clarity=memory["clarity"],
        )
        reset = self.circuits.measure_reset(
            overload=load_temperature,
            contradiction=reflection.contradiction,
            evidence=reflection.evidence,
            memory_pressure=memory["fusion"],
        )
        if reset["trigger"] > 0.74:
            phase = self.reset_phases[0 if reset["clear"] >= reset["recollect"] else 1]
            if reset["rebuild"] > max(reset["clear"], reset["recollect"]):
                phase = self.reset_phases[2]
            if reset["resume"] > 0.82:
                phase = self.reset_phases[3]
        else:
            phase = self.reset_phases[3]

        if chosen_color in {"Crimson", "Violet", "Obsidian"}:
            selected_band = "ultraviolet"
        elif chosen_color in {"Gold", "Jade"}:
            selected_band = "infrared"

        aligned = list(dict.fromkeys([
            *ACTIVE_RUNTIME_PRIMITIVES["Perception"],
            *ACTIVE_RUNTIME_PRIMITIVES["Planning"],
            *ACTIVE_RUNTIME_PRIMITIVES["Memory/feedback"],
            *ACTIVE_RUNTIME_PRIMITIVES["Coordination"],
            *ACTIVE_RUNTIME_PRIMITIVES["Oversight"],
            *self._select_aligned_concepts(domain, selected_band, selected_task, load_temperature),
        ]))
        memory_echo = self._update_memory_bank(scenario_key, selected_task, mood_label, aligned)
        temporal_pattern = self._score_temporal_pattern(scenario_key, chosen_color)
        penalties = self._penalty_metrics(
            scenario_key=scenario_key,
            chosen_color=chosen_color,
            color_state=color_state,
            attention=attention,
            reset=reset,
            temporal_pattern=temporal_pattern,
            domain_signature=domain_signature,
        )
        color_audit = self._build_color_audit(
            chosen_color=chosen_color,
            color_state=color_state,
            domain_signature=domain_signature,
            selected_task=selected_task,
            ranking=ranking,
        )
        total_penalty = (
            penalties["ambiguity_persistence"]
            + penalties["contradiction_debt"] * 0.15
            + penalties["delayed_recovery_cost"]
            + penalties["false_calm_penalty"]
            + penalties["reset_misuse_penalty"]
            + penalties["temporal_penalty"]
            - penalties["temporal_bonus"]
        )
        state_deltas = {
            "risk_pressure": -0.030 * selected["metrics"]["safety"] + 0.016 * attention["conflict"] + total_penalty * 0.18,
            "system_stability": 0.022 * confidence["stability"] + 0.014 * rewards["safety"] - total_penalty * 0.14,
            "human_operator_load": -0.018 * rewards["safety"] + 0.030 * load_temperature + penalties["delayed_recovery_cost"] * 0.4,
            "environmental_chaos": -0.015 * memory["clarity"] + 0.010 * attention["monitor"] + penalties["false_calm_penalty"] * 0.25,
            "sensor_conflict": -0.020 * confidence["evidence"] + 0.020 * attention["conflict"] + penalties["ambiguity_persistence"] * 0.35,
            "route_coherence": 0.018 * schedule["visible"] + 0.014 * selected["metrics"]["balance"] - penalties["contradiction_debt"] * 0.06,
            "mechanical_stress": -0.016 * rewards["safety"] + 0.016 * schedule["infrared"] + penalties["reset_misuse_penalty"] * 0.20,
            "intervention_readiness": 0.024 * safe_mean((memory["fusion"], confidence["depth"], rewards["safety"])) - total_penalty * 0.08,
        }
        processor_metrics = {
            "intent_coherence": intent_metrics["coherence"],
            "task_balance": selected["metrics"]["balance"],
            "schedule_mix": schedule["mix"],
            "memory_fusion": memory["fusion"],
            "reward_interference": rewards["interference"],
            "attention_focus": attention["focus"],
            "confidence_depth": confidence["depth"],
            "reset_trigger": reset["trigger"],
            "domain_pressure": domain_signature["composite"],
            "total_penalty": total_penalty,
        }
        return AdvancedLoopResult(
            scenario_key=scenario_key,
            month=month,
            intent_palette=self.palette_engine.classify_temperature(intent_metrics["urgency"]),
            selected_task=selected_task.title,
            selected_band=selected_band,
            chosen_color=chosen_color,
            mood_label=mood_label,
            confidence_depth=self.palette_engine.depth_label(confidence["depth"]),
            load_temperature=load_temperature,
            color_state=color_state,
            bloom_focus=attention,
            reward_channels=rewards,
            reflection_echo={
                "brightness": reflection.brightness,
                "contradiction": reflection.contradiction,
                "evidence": reflection.evidence,
                "clarity": reflection.clarity,
            },
            reset_signal={
                "phase": phase.name,
                "template": phase.template,
                "palette_anchor": phase.palette_anchor,
                "trigger": reset["trigger"],
            },
            memory_echo=memory_echo,
            concept_alignment=aligned,
            active_primitives={family: list(names) for family, names in ACTIVE_RUNTIME_PRIMITIVES.items()},
            domain_signature=domain_signature,
            temporal_pattern=temporal_pattern,
            color_audit=color_audit,
            penalty_metrics=penalties,
            state_deltas=state_deltas,
            processor_metrics=processor_metrics,
            task_ranking=[
                {
                    "task": item["task"].title,
                    "runtime_primitive": item["task"].runtime_primitive,
                    "authority_zone": item["task"].authority_zone,
                    "composite": item["composite"],
                    "competition_penalty": item.get("competition_penalty", 0.0),
                    "priority": item["metrics"]["priority"],
                    "safety": item["metrics"]["safety"],
                    "novelty": item["metrics"]["novelty"],
                }
                for item in ranking[:5]
            ],
        )


def build_advanced_color_agentic_loop_system() -> AdvancedColorAgenticLoopSystem:
    return AdvancedColorAgenticLoopSystem()


# Cell 2 - Scenario bank, advanced blog concepts, and generation blueprint

DEFAULT_SCENARIO_BANK = {
    "advanced_blog": [
        "Urban intersection collision forecasting with entropic quantum traffic intelligence",
        "Highway multi-vehicle crash prediction using uncertainty-aware mobility modeling",
        "Maritime shipwreck prevention through wave entropy and navigation coherence analysis",
        "Aircraft instability forecasting using turbulence entropy, sensor disagreement, and predictive aviation AI",
        "City-scale transportation safety intelligence for autonomous and human-driven systems",
    ]
}

BLOG_BLUEPRINT = {
    "title_patterns": [
        "How Entropic Quantum Intelligence Could Transform Transportation Safety",
        "From Traffic Collisions to Shipwreck Prevention: The Rise of Quantum-Inspired Safety AI",
        "Predicting Crashes Before They Happen: Advanced AI Simulation for Roads, Ships, and Aircraft",
    ],
    "sections": [
        "Introduction",
        "Why modern safety prediction needs a new intelligence model",
        "What entropic quantum intelligence means in practical terms",
        "Road traffic safety intelligence and real-world accident prediction",
        "Shipwreck forecasting and maritime instability mapping",
        "Airplane crash precursor detection and aviation intelligence",
        "Invented next-generation concepts for safety forecasting",
        "Simulation results and what they suggest",
        "Why uncertainty-aware AI matters more than raw prediction accuracy",
        "Ethics, limitations, and deployment challenges",
        "The future of predictive safety intelligence",
        "Conclusion",
    ],
    "seo_keywords": [
        "AI traffic safety intelligence",
        "car accident prediction AI",
        "entropic quantum intelligence",
        "shipwreck prediction AI",
        "airplane crash prediction",
        "predictive safety simulation",
        "quantum-inspired transportation AI",
        "safety intelligence systems",
    ],
}

ADVANCED_AGENTIC_SYSTEM = build_advanced_color_agentic_loop_system()
ADVANCED_CONCEPT_BANK = ADVANCED_AGENTIC_SYSTEM.export_concept_bank()
AGENTIC_SURFACE_FAMILIES = ADVANCED_AGENTIC_SYSTEM.export_family_map()
COLOR_QUANTUM_ENCODING = ADVANCED_AGENTIC_SYSTEM.export_encoding_notes()
CONTROL_PALETTE = ADVANCED_AGENTIC_SYSTEM.export_palette()
ADVANCED_COLOR_LOOP_SEQUENCE = ADVANCED_AGENTIC_SYSTEM.ordered_advanced_concept_names()
EXPANSION_CONCEPT_SEQUENCE = ADVANCED_AGENTIC_SYSTEM.ordered_expansion_concept_names()
FOUNDATIONAL_CONCEPT_SEQUENCE = ADVANCED_AGENTIC_SYSTEM.ordered_foundational_concept_names()

SELECTED_SCENARIOS = list(DEFAULT_SCENARIO_BANK["advanced_blog"])


# Cell 3 - Helper utilities, memory, summarization, structure, and writing tools

def debug_print(tag: str, message: str) -> None:
    if DEBUG_VERBOSE:
        print(f"[DEBUG] {tag} | {message}")


def reflow(text: str) -> str:
    return " ".join(str(text).split())


def short_text(text: str, limit: int = 180) -> str:
    text = reflow(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def hashed_embedding(text: str, dims: int = 72) -> List[float]:
    vec = [0.0] * dims
    tokens = [tok.lower() for tok in reflow(text).split() if tok.strip()]
    if not tokens:
        return vec
    for token in tokens:
        h = stable_hash(token)
        for i in range(0, min(len(h), dims * 2), 2):
            idx = (i // 2) % dims
            val = (int(h[i:i+2], 16) / 255.0) - 0.5
            vec[idx] += val
    scale = max(1, len(tokens))
    return [v / scale for v in vec]


def ensure_memory_db() -> None:
    if not USE_SQLITE_MEMORY:
        return
    conn = sqlite3.connect(MEMORY_DB_PATH)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS blog_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                topic TEXT NOT NULL,
                scenario TEXT NOT NULL,
                summary TEXT NOT NULL,
                score REAL NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS blog_fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                topic TEXT NOT NULL,
                kind TEXT NOT NULL,
                text_fragment TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                salience REAL NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def store_fragment(topic: str, kind: str, text_fragment: str, salience: float) -> None:
    ensure_memory_db()
    if not USE_SQLITE_MEMORY:
        return
    conn = sqlite3.connect(MEMORY_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO blog_fragments(created_at, topic, kind, text_fragment, embedding_json, salience)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                topic,
                kind,
                text_fragment,
                json.dumps(hashed_embedding(text_fragment)),
                float(salience),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def retrieve_fragments(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    ensure_memory_db()
    if not USE_SQLITE_MEMORY:
        return []

    qvec = hashed_embedding(query)
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT topic, kind, text_fragment, embedding_json, salience FROM blog_fragments ORDER BY id DESC LIMIT 500"
        ).fetchall()
    finally:
        conn.close()

    scored = []
    for row in rows:
        emb = json.loads(row["embedding_json"])
        sim = cosine_similarity(qvec, emb)
        score = 0.7 * sim + 0.3 * float(row["salience"])
        scored.append(
            {
                "topic": row["topic"],
                "kind": row["kind"],
                "text_fragment": row["text_fragment"],
                "score": score,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def store_run(topic: str, scenario: str, summary: str, score: float, payload: Dict[str, Any]) -> None:
    ensure_memory_db()
    if not USE_SQLITE_MEMORY:
        return
    conn = sqlite3.connect(MEMORY_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO blog_runs(created_at, topic, scenario, summary, score, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (time.time(), topic, scenario, summary, float(score), json.dumps(payload)),
        )
        conn.commit()
    finally:
        conn.close()


def load_llm() -> Optional[Any]:
    global LLM
    if not ENABLE_LLM_SUMMARY:
        return None
    if LLM is not None:
        return LLM
    if Llama is None:
        debug_print("llm", "llama_cpp unavailable")
        return None
    if not GGUF_PATH or not os.path.exists(GGUF_PATH):
        debug_print("llm", f"GGUF not found at {GGUF_PATH}")
        return None
    LLM = Llama(
        model_path=GGUF_PATH,
        n_ctx=int(CTX_SIZE),
        n_threads=int(THREADS),
        n_gpu_layers=int(N_GPU_LAYERS),
        verbose=False,
    )
    return LLM


def trim_prompt(text: str, limit_chars: int = 12000) -> str:
    text = text.strip()
    if len(text) <= limit_chars:
        return text
    head = text[: limit_chars // 2]
    tail = text[-limit_chars // 2 :]
    return head + "\n[... trimmed ...]\n" + tail


def llm_complete(prompt: str) -> str:
    model = load_llm()
    if model is None:
        return ""
    out = model(
        trim_prompt(prompt),
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.92,
        repeat_penalty=1.08,
        stop=["</end>", "\n\n\n\n"],
    )
    return out["choices"][0]["text"].strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def sentence_split(text: str) -> List[str]:
    if nltk_sent_tokenize is not None:
        try:
            return [reflow(s) for s in nltk_sent_tokenize(text) if reflow(s)]
        except Exception:
            pass
    rough = re.split(r"(?<=[.!?])\s+", reflow(text))
    return [s for s in rough if s.strip()]


def paragraphize(sentences: List[str], min_sents: int = 4, max_sents: int = 7) -> str:
    out = []
    cursor = 0
    rng = random.Random(42)
    while cursor < len(sentences):
        block = rng.randint(min_sents, max_sents)
        piece = " ".join(sentences[cursor: cursor + block]).strip()
        if piece:
            out.append(piece)
        cursor += block
    return "\n\n".join(out)


def keyword_surface(text: str) -> List[str]:
    if summa_keywords is not None:
        try:
            kws = summa_keywords.keywords(text, words=12, split=True)
            if kws:
                return [str(k) for k in kws][:12]
        except Exception:
            pass
    tokens = [tok.lower() for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]{4,}", text)]
    freq = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:12]]


def summarize_surface(text: str, limit: int = 8) -> List[str]:
    if summa_summarizer is not None:
        try:
            out = summa_summarizer.summarize(text, split=True)
            if out:
                return [reflow(x) for x in out[:limit]]
        except Exception:
            pass
    return sentence_split(text)[:limit]


def choose_title() -> str:
    return random.choice(BLOG_BLUEPRINT["title_patterns"])


def concept_lookup() -> Dict[str, Dict[str, str]]:
    return {concept["name"]: concept for concept in ADVANCED_CONCEPT_BANK}


def ordered_advanced_concepts() -> List[Dict[str, str]]:
    lookup = concept_lookup()
    return [lookup[name] for name in ADVANCED_COLOR_LOOP_SEQUENCE if name in lookup]


def validate_advanced_concepts() -> Dict[str, Any]:
    return ADVANCED_AGENTIC_SYSTEM.validate_primary_concepts()


def choose_concepts(n: int = 5) -> List[Dict[str, str]]:
    pool = ADVANCED_CONCEPT_BANK[:]
    return pool[: min(n, len(pool))]


def generate_meta_description(title: str, topic: str) -> str:
    raw = (
        f"{title}. Explore how advanced AI simulations, entropic quantum intelligence, and "
        f"uncertainty-aware safety systems could help predict road crashes, shipwrecks, and aviation failures."
    )
    return raw[:156]


def generate_blog_outline(title: str, scenarios: List[str], concepts: List[Dict[str, str]]) -> List[str]:
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Outline")
    for idx, section in enumerate(BLOG_BLUEPRINT["sections"], 1):
        lines.append(f"{idx}. {section}")
    lines.append("")
    lines.append("## Scenario anchors")
    for item in scenarios:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Invented concepts to develop")
    for c in concepts:
        lines.append(f"- {c['name']}: {c['tagline']}")
    lines.append("")
    validation = validate_advanced_concepts()
    lines.append("## Validated advanced concepts one by one")
    for idx, name in enumerate(validation["ordered_names"], 1):
        lines.append(f"{idx}. {name}")
    lines.append("")
    lines.append("## Expansion concept bank")
    for idx, name in enumerate(EXPANSION_CONCEPT_SEQUENCE, 1):
        lines.append(f"{idx}. {name}")
    lines.append("")
    lines.append("## Foundational concept bank")
    for idx, name in enumerate(FOUNDATIONAL_CONCEPT_SEQUENCE, 1):
        lines.append(f"{idx}. {name}")
    if validation["missing"]:
        lines.append(f"Missing: {', '.join(validation['missing'])}")
    if validation["duplicates"]:
        lines.append(f"Duplicates: {', '.join(validation['duplicates'])}")
    lines.append("")
    lines.append("## Agentic loop surface families")
    for family, members in AGENTIC_SURFACE_FAMILIES.items():
        lines.append(f"- {family}: {', '.join(members)}")
    return lines


def expand_concept_block(concept: Dict[str, str]) -> str:
    return (
        f"### {concept['name']}\n\n"
        f"**Core idea:** {concept['tagline']}\n\n"
        f"{concept['explanation']}\n\n"
        f"In the context of an advanced blog generator, this concept can be used as a framework for explaining "
        f"how AI does more than score danger. It maps invisible instability, translates ambiguity into structured "
        f"signals, and helps readers imagine how next-generation civilian safety systems may operate across roads, "
        f"shipping routes, and aircraft operations."
    )


def expand_numbered_concept_block(index: int, concept: Dict[str, str]) -> str:
    return expand_concept_block(concept).replace("### ", f"### {index}. ", 1)


def repeat_to_word_target(text: str, target_words: int) -> str:
    if count_words(text) >= target_words:
        return text
    sentences = sentence_split(text)
    if not sentences:
        return text
    out = [text]
    idx = 0
    while count_words("\n\n".join(out)) < target_words:
        s = sentences[idx % len(sentences)]
        out.append(
            f"{s} This matters because safety intelligence becomes most valuable when it can act before visible failure, "
            f"not after damage is already underway."
        )
        idx += 1
        if idx > 2000:
            break
    return "\n\n".join(out)

# Cell 4 - Advanced entropic quantum safety simulation and 7000-word blog generator

# -------------------------
# Quantum device
# -------------------------
QDEV = qml.device("default.qubit", wires=6)


@qml.qnode(QDEV)
def quantum_safety_surface(seed_angles: List[float]):
    for wire, angle in enumerate(seed_angles[:6]):
        qml.RY(angle, wires=wire)
        qml.RZ(angle * 0.73, wires=wire)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 5])
    qml.CRX(seed_angles[0] * 0.4, wires=[0, 3])
    qml.CRY(seed_angles[1] * 0.5, wires=[1, 4])
    qml.CRZ(seed_angles[2] * 0.6, wires=[2, 5])
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]


# -------------------------
# Data classes
# -------------------------
@dataclass
class SafetySignal:
    name: str
    value: float
    description: str


@dataclass
class SafetyScenario:
    scenario: str
    domain: str
    baseline_risk: float
    signals: List[SafetySignal] = field(default_factory=list)


@dataclass
class ScenarioState:
    scenario: str
    domain: str
    month: int
    risk_pressure: float
    system_stability: float
    human_operator_load: float
    environmental_chaos: float
    sensor_conflict: float
    route_coherence: float
    mechanical_stress: float
    intervention_readiness: float
    notes: List[str] = field(default_factory=list)
    applied_actions: List[str] = field(default_factory=list)


@dataclass
class SafetyIntervention:
    name: str
    stabilization_weight: float
    sensor_weight: float
    route_weight: float
    maintenance_weight: float
    human_weight: float
    explanation: str


@dataclass
class SimulationResult:
    scenario: str
    domain: str
    run_id: str
    score: float
    road_risk_score: float
    ship_risk_score: float
    aviation_risk_score: float
    coherence_score: float
    intervention_score: float
    summary: str
    timeline: List[Dict[str, Any]]
    applied_path: List[str]


# -------------------------
# Intervention library
# -------------------------
SAFETY_INTERVENTIONS = [
    SafetyIntervention(
        "Adaptive speed moderation layer",
        0.91, 0.44, 0.72, 0.31, 0.65,
        "Uses real-time instability forecasting to reduce velocity before collision cascades form."
    ),
    SafetyIntervention(
        "Sensor confidence arbitration",
        0.62, 0.95, 0.46, 0.28, 0.51,
        "Resolves disagreement between cameras, radar, lidar, vibration streams, and environmental instruments."
    ),
    SafetyIntervention(
        "Route coherence rebalance",
        0.74, 0.52, 0.96, 0.34, 0.43,
        "Recomputes safer pathing when drift, congestion, sea-state instability, or corridor turbulence emerges."
    ),
    SafetyIntervention(
        "Predictive maintenance sentinel",
        0.68, 0.39, 0.41, 0.98, 0.33,
        "Detects early stress signatures in braking systems, engines, hull integrity, or aircraft subsystems."
    ),
    SafetyIntervention(
        "Human-machine attention relief",
        0.56, 0.36, 0.37, 0.21, 0.97,
        "Reduces overload by timing alerts, simplifying guidance, and filtering noise during high-risk moments."
    ),
    SafetyIntervention(
        "Entropic weather compensation",
        0.79, 0.58, 0.66, 0.42, 0.49,
        "Applies uncertainty-aware correction when rain, fog, crosswinds, wave conditions, or icing increase chaos."
    ),
    SafetyIntervention(
        "Recursive sentinel re-evaluation",
        0.64, 0.86, 0.53, 0.47, 0.69,
        "Continuously questions the model's confidence under missing, conflicting, or degraded data."
    ),
    SafetyIntervention(
        "Failure echo anomaly watch",
        0.73, 0.67, 0.61, 0.77, 0.44,
        "Detects weak pre-failure patterns before they become visible emergencies."
    ),
]


# -------------------------
# Scenario builders
# -------------------------
def classify_domain(scenario: str) -> str:
    s = scenario.lower()
    if "ship" in s or "maritime" in s or "wave" in s:
        return "maritime"
    if "aircraft" in s or "airplane" in s or "aviation" in s or "flight" in s:
        return "aviation"
    return "road"


def build_signals_for_domain(domain: str) -> List[SafetySignal]:
    if domain == "road":
        return [
            SafetySignal("traffic_density", 0.71, "Congestion and interaction pressure among vehicles"),
            SafetySignal("speed_variance", 0.67, "Unstable speed spread raises collision probability"),
            SafetySignal("weather_entropy", 0.49, "Rain, glare, fog, or surface unpredictability"),
            SafetySignal("driver_reaction_latency", 0.54, "Delayed braking or steering response"),
            SafetySignal("sensor_uncertainty", 0.42, "Camera/radar ambiguity"),
            SafetySignal("road_surface_instability", 0.46, "Potholes, debris, ice, or degraded pavement"),
        ]
    if domain == "maritime":
        return [
            SafetySignal("wave_entropy", 0.72, "Chaotic sea-state energy and directional inconsistency"),
            SafetySignal("navigation_drift", 0.51, "Route deviation under stress"),
            SafetySignal("engine_stress", 0.48, "Mechanical strain and propulsion inconsistency"),
            SafetySignal("crew_fatigue", 0.56, "Human attention degradation"),
            SafetySignal("visibility_instability", 0.45, "Fog, darkness, storm interference"),
            SafetySignal("hull_strain_signal", 0.44, "Structural stress accumulation"),
        ]
    return [
        SafetySignal("turbulence_entropy", 0.69, "Atmospheric instability and sudden motion risk"),
        SafetySignal("sensor_disagreement", 0.47, "Conflicting avionics or instrument interpretations"),
        SafetySignal("engine_vibration_variance", 0.43, "Mechanical stress fluctuation"),
        SafetySignal("navigation_corridor_drift", 0.39, "Deviation from stable operational path"),
        SafetySignal("icing_or_weather_complexity", 0.52, "Environmental hazard layering"),
        SafetySignal("crew_attention_load", 0.41, "Operational and cognitive demand"),
    ]


def build_initial_scenario(scenario: str) -> SafetyScenario:
    domain = classify_domain(scenario)
    baseline = {"road": 0.63, "maritime": 0.59, "aviation": 0.57}[domain]
    return SafetyScenario(
        scenario=scenario,
        domain=domain,
        baseline_risk=baseline,
        signals=build_signals_for_domain(domain),
    )


def build_initial_state(scenario: str) -> ScenarioState:
    sc = build_initial_scenario(scenario)
    return ScenarioState(
        scenario=scenario,
        domain=sc.domain,
        month=0,
        risk_pressure=sc.baseline_risk,
        system_stability=0.41,
        human_operator_load=0.56,
        environmental_chaos=0.52,
        sensor_conflict=0.44,
        route_coherence=0.46,
        mechanical_stress=0.43,
        intervention_readiness=0.39,
        notes=[f"Initialized scenario for domain={sc.domain}"],
        applied_actions=[],
    )


# -------------------------
# Entropy and quantum metrics
# -------------------------
def entropy_snapshot(scenario: str, run_index: int, domain: str) -> Dict[str, float]:
    ts = time.time()
    cpu = psutil.cpu_percent(interval=0.05) / 100.0
    ram = psutil.virtual_memory().percent / 100.0
    h = stable_hash(f"{scenario}|{run_index}|{domain}|{ts:.5f}")
    raw = [int(h[i:i+2], 16) / 255.0 for i in range(0, 12, 2)]
    return {
        "cpu": cpu,
        "ram": ram,
        "hash_entropy": statistics.fmean(raw),
        "time_wave": abs(math.sin(ts / 60.0)),
        "weather_noise": raw[0],
        "signal_noise": raw[1],
        "motion_noise": raw[2],
        "operator_noise": raw[3],
        "route_noise": raw[4],
        "stress_noise": raw[5],
    }


def build_quantum_metrics(snapshot: Dict[str, float]) -> Dict[str, float]:
    angles = [
        snapshot["cpu"] * math.pi,
        snapshot["ram"] * math.pi,
        snapshot["hash_entropy"] * math.pi,
        snapshot["time_wave"] * math.pi,
        snapshot["weather_noise"] * math.pi,
        snapshot["signal_noise"] * math.pi,
    ]
    vec = [float(x) for x in quantum_safety_surface(angles)]
    metrics = {
        "stability": (vec[0] + 1.0) / 2.0,
        "coherence": (vec[1] + 1.0) / 2.0,
        "warning_clarity": (vec[2] + 1.0) / 2.0,
        "route_integrity": (vec[3] + 1.0) / 2.0,
        "mechanical_resilience": (vec[4] + 1.0) / 2.0,
        "attention_balance": (vec[5] + 1.0) / 2.0,
    }
    metrics["field_strength"] = statistics.fmean(list(metrics.values()))
    return metrics


def choose_interventions(snapshot: Dict[str, float], qmetrics: Dict[str, float], k: int = 3) -> List[SafetyIntervention]:
    scored = []
    for item in SAFETY_INTERVENTIONS:
        score = (
            0.20 * item.stabilization_weight * qmetrics["stability"] +
            0.18 * item.sensor_weight * qmetrics["warning_clarity"] +
            0.18 * item.route_weight * qmetrics["route_integrity"] +
            0.17 * item.maintenance_weight * qmetrics["mechanical_resilience"] +
            0.17 * item.human_weight * qmetrics["attention_balance"] +
            0.10 * (1.0 - snapshot["signal_noise"] * 0.4)
        )
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


def apply_intervention(state: ScenarioState, intervention: SafetyIntervention, qmetrics: Dict[str, float], snapshot: Dict[str, float]) -> None:
    entropy_push = snapshot["hash_entropy"] * 0.05
    state.risk_pressure = clamp(
        state.risk_pressure - 0.08 * intervention.stabilization_weight * qmetrics["stability"] + 0.02 * entropy_push
    )
    state.system_stability = clamp(
        state.system_stability + 0.09 * intervention.stabilization_weight * qmetrics["coherence"]
    )
    state.sensor_conflict = clamp(
        state.sensor_conflict - 0.09 * intervention.sensor_weight * qmetrics["warning_clarity"] + 0.01 * snapshot["signal_noise"]
    )
    state.route_coherence = clamp(
        state.route_coherence + 0.11 * intervention.route_weight * qmetrics["route_integrity"]
    )
    state.mechanical_stress = clamp(
        state.mechanical_stress - 0.07 * intervention.maintenance_weight * qmetrics["mechanical_resilience"] + 0.01 * snapshot["stress_noise"]
    )
    state.human_operator_load = clamp(
        state.human_operator_load - 0.08 * intervention.human_weight * qmetrics["attention_balance"] + 0.02 * snapshot["operator_noise"]
    )
    state.environmental_chaos = clamp(
        state.environmental_chaos - 0.04 * qmetrics["coherence"] + 0.03 * snapshot["weather_noise"]
    )
    state.intervention_readiness = clamp(
        state.intervention_readiness + 0.08 * qmetrics["field_strength"]
    )

    state.applied_actions.append(intervention.name)
    state.notes.append(intervention.explanation)


def monthly_drift(state: ScenarioState, snapshot: Dict[str, float], qmetrics: Dict[str, float]) -> None:
    drift = 0.012 + snapshot["time_wave"] * 0.015
    state.risk_pressure = clamp(
        state.risk_pressure + drift + 0.015 * state.environmental_chaos - 0.03 * state.intervention_readiness
    )
    state.system_stability = clamp(
        state.system_stability - 0.02 * state.risk_pressure + 0.015 * qmetrics["coherence"]
    )
    state.sensor_conflict = clamp(
        state.sensor_conflict + 0.01 * snapshot["signal_noise"] - 0.02 * qmetrics["warning_clarity"]
    )
    state.route_coherence = clamp(
        state.route_coherence - 0.015 * state.environmental_chaos + 0.02 * qmetrics["route_integrity"]
    )
    state.mechanical_stress = clamp(
        state.mechanical_stress + 0.012 * snapshot["stress_noise"] - 0.015 * qmetrics["mechanical_resilience"]
    )
    state.human_operator_load = clamp(
        state.human_operator_load + 0.015 * state.risk_pressure - 0.02 * qmetrics["attention_balance"]
    )


def apply_advanced_loop_feedback(state: ScenarioState, loop_result: Any) -> None:
    for field_name, delta in loop_result.state_deltas.items():
        current = getattr(state, field_name)
        setattr(state, field_name, clamp(current + float(delta)))
    state.applied_actions.append(f"Agentic::{loop_result.selected_task}")
    state.notes.append(
        f"Advanced loop chose color={loop_result.chosen_color}, task='{loop_result.selected_task}', band={loop_result.selected_band}, mood={loop_result.mood_label}, depth={loop_result.confidence_depth}, reset={loop_result.reset_signal['phase']}"
    )
    state.notes.append(
        f"Color audit: {loop_result.color_audit['reason']} | next={', '.join(loop_result.color_audit['next_allowed_transitions'][:3])}"
    )
    if loop_result.temporal_pattern.get('label') and loop_result.temporal_pattern['label'] != 'none':
        state.notes.append(f"Temporal pattern: {loop_result.temporal_pattern['label']}")
    if loop_result.concept_alignment:
        state.notes.append("Aligned concepts: " + ", ".join(loop_result.concept_alignment[:4]))


def score_state(state: ScenarioState) -> Dict[str, float]:
    road_risk = 1.0 - state.risk_pressure if state.domain == "road" else 0.5 + (1.0 - state.risk_pressure) * 0.5
    ship_risk = 1.0 - state.risk_pressure if state.domain == "maritime" else 0.5 + (1.0 - state.risk_pressure) * 0.5
    aviation_risk = 1.0 - state.risk_pressure if state.domain == "aviation" else 0.5 + (1.0 - state.risk_pressure) * 0.5
    coherence = statistics.fmean([
        state.system_stability,
        state.route_coherence,
        1.0 - state.sensor_conflict,
        1.0 - state.human_operator_load,
        1.0 - state.mechanical_stress,
    ])
    intervention_score = state.intervention_readiness
    total = statistics.fmean([road_risk, ship_risk, aviation_risk, coherence, intervention_score])
    return {
        "score": total,
        "road_risk_score": road_risk,
        "ship_risk_score": ship_risk,
        "aviation_risk_score": aviation_risk,
        "coherence_score": coherence,
        "intervention_score": intervention_score,
    }


def deterministic_result_summary(state: ScenarioState, scores: Dict[str, float]) -> str:
    last_notes = "; ".join(state.notes[-5:])
    return (
        f"Scenario '{state.scenario}' in domain '{state.domain}' reached a composite safety score of {scores['score']:.3f}. "
        f"Road={scores['road_risk_score']:.3f}, maritime={scores['ship_risk_score']:.3f}, aviation={scores['aviation_risk_score']:.3f}, "
        f"coherence={scores['coherence_score']:.3f}, intervention_readiness={scores['intervention_score']:.3f}. "
        f"Recent interpretation: {last_notes}"
    )


def simulate_path(scenario: str, run_index: int, months: int) -> SimulationResult:
    state = build_initial_state(scenario)
    timeline = []
    for month in range(1, months + 1):
        snapshot = entropy_snapshot(scenario, run_index * 100 + month, state.domain)
        qmetrics = build_quantum_metrics(snapshot)
        interventions = choose_interventions(snapshot, qmetrics, k=2 if month % 2 else 3)
        for item in interventions:
            apply_intervention(state, item, qmetrics, snapshot)
        loop_result = ADVANCED_AGENTIC_SYSTEM.process_cycle(
            scenario=scenario,
            domain=state.domain,
            month=month,
            snapshot=snapshot,
            qmetrics=qmetrics,
            state={
                "risk_pressure": state.risk_pressure,
                "system_stability": state.system_stability,
                "human_operator_load": state.human_operator_load,
                "environmental_chaos": state.environmental_chaos,
                "sensor_conflict": state.sensor_conflict,
                "route_coherence": state.route_coherence,
                "mechanical_stress": state.mechanical_stress,
                "intervention_readiness": state.intervention_readiness,
            },
            interventions=[x.name for x in interventions],
        )
        apply_advanced_loop_feedback(state, loop_result)
        monthly_drift(state, snapshot, qmetrics)
        state.month = month
        timeline.append(
            {
                "month": month,
                "snapshot": snapshot,
                "qmetrics": qmetrics,
                "risk_pressure": state.risk_pressure,
                "system_stability": state.system_stability,
                "human_operator_load": state.human_operator_load,
                "environmental_chaos": state.environmental_chaos,
                "sensor_conflict": state.sensor_conflict,
                "route_coherence": state.route_coherence,
                "mechanical_stress": state.mechanical_stress,
                "intervention_readiness": state.intervention_readiness,
                "interventions": [x.name for x in interventions],
                "advanced_loop": asdict(loop_result),
            }
        )
    scores = score_state(state)
    summary = deterministic_result_summary(state, scores)
    return SimulationResult(
        scenario=scenario,
        domain=state.domain,
        run_id=f"run::{stable_hash(scenario + str(run_index))[:12]}",
        score=scores["score"],
        road_risk_score=scores["road_risk_score"],
        ship_risk_score=scores["ship_risk_score"],
        aviation_risk_score=scores["aviation_risk_score"],
        coherence_score=scores["coherence_score"],
        intervention_score=scores["intervention_score"],
        summary=summary,
        timeline=timeline,
        applied_path=state.applied_actions,
    )


def aggregate_results(results: List[SimulationResult]) -> Dict[str, Any]:
    ranked = sorted(results, key=lambda x: x.score, reverse=True)
    top = ranked[:TOP_K_PATHS]
    all_summaries = "\n".join(x.summary for x in top)
    intervention_frequency = {}
    band_frequency = {}
    reset_frequency = {}
    color_frequency = {}
    temporal_pattern_frequency = {}
    concept_frequency = {}
    load_temperatures = []
    penalty_totals = []
    for item in top:
        for action in item.applied_path:
            intervention_frequency[action] = intervention_frequency.get(action, 0) + 1
        for step in item.timeline:
            loop = step.get("advanced_loop", {})
            if not loop:
                continue
            band = loop.get("selected_band")
            reset_phase = loop.get("reset_signal", {}).get("phase")
            chosen_color = loop.get("chosen_color")
            pattern_label = loop.get("temporal_pattern", {}).get("label")
            if band:
                band_frequency[band] = band_frequency.get(band, 0) + 1
            if reset_phase:
                reset_frequency[reset_phase] = reset_frequency.get(reset_phase, 0) + 1
            if chosen_color:
                color_frequency[chosen_color] = color_frequency.get(chosen_color, 0) + 1
            if pattern_label and pattern_label != "none":
                temporal_pattern_frequency[pattern_label] = temporal_pattern_frequency.get(pattern_label, 0) + 1
            if loop.get("load_temperature") is not None:
                load_temperatures.append(float(loop["load_temperature"]))
            if loop.get("processor_metrics", {}).get("total_penalty") is not None:
                penalty_totals.append(float(loop["processor_metrics"]["total_penalty"]))
            for concept_name in loop.get("concept_alignment", []):
                concept_frequency[concept_name] = concept_frequency.get(concept_name, 0) + 1

    domain_mix = {"road": 0, "maritime": 0, "aviation": 0}
    for item in results:
        domain_mix[item.domain] += 1

    return {
        "avg_score": statistics.fmean([x.score for x in results]) if results else 0.0,
        "top_results": [asdict(x) for x in top],
        "intervention_frequency": intervention_frequency,
        "agentic_band_frequency": band_frequency,
        "reset_phase_frequency": reset_frequency,
        "chosen_color_frequency": color_frequency,
        "temporal_pattern_frequency": temporal_pattern_frequency,
        "concept_alignment_frequency": concept_frequency,
        "avg_agentic_load": statistics.fmean(load_temperatures) if load_temperatures else 0.0,
        "avg_penalty_pressure": statistics.fmean(penalty_totals) if penalty_totals else 0.0,
        "keyword_surface": keyword_surface(all_summaries),
        "sentence_surface": summarize_surface(all_summaries, limit=10),
        "memory_hits": retrieve_fragments(BLOG_TOPIC, limit=8),
        "domain_mix": domain_mix,
    }


# -------------------------
# Blog writing engine
# -------------------------
def intro_section(title: str) -> str:
    return f"""
## Introduction

{title} is not just a futuristic slogan. It names a new class of thinking about risk, one in which accidents are no longer treated as isolated surprises. Instead, crashes, shipwrecks, and aircraft emergencies are viewed as the visible outcomes of hidden instability that builds across time. Modern transportation already produces immense streams of information, yet much of that information is fragmented, delayed, or interpreted too narrowly. Cameras watch roads, radars scan distance, aircraft instruments monitor altitude and velocity, and ships track routes and engine status. But conventional systems often interpret these signals as separate channels rather than as interacting fields of uncertainty.

That limitation matters because real-world failures usually form as cascades. A traffic collision can begin with congestion pressure, degraded road conditions, weather irregularity, small reaction delays, and subtle sensor ambiguity. A shipwreck may not begin at the moment of impact, but much earlier when route coherence weakens under conflicting currents, stress accumulates in the hull, visibility degrades, and operator fatigue rises. An aviation emergency may similarly emerge from a chain of turbulence, instrument disagreement, mechanical vibration, corridor drift, and growing cockpit workload. When viewed in isolation, each signal may seem survivable. When entangled, those signals create a dangerous geometry of risk.

That is why this notebook redesign introduces the idea of entropic quantum intelligence. The phrase does not claim literal magical prediction. Instead, it describes an advanced simulation framework that uses entropy-like measurements, uncertainty surfaces, route memory, and quantum-inspired transformations to model how instability behaves before disaster becomes obvious. The goal is prevention, not spectacle. The goal is earlier awareness, better warnings, smarter interventions, and richer public understanding of how predictive safety intelligence could evolve over the coming years.

This blog explores how such a system could work across road traffic, maritime navigation, and aviation operations. It also introduces new invented concepts designed for long-form technical storytelling: the Entropic Quantum Safety Field, the Predictive Fracture Horizon, the Causal Turbulence Index, the Recursive Sentinel Layer, Quantum Route Memory, Failure Echo Mapping, and the Safety Coherence Gradient. Together, these concepts form the intellectual backbone of a next-generation blog generator capable of turning simulation results into a substantial, readable, and concept-rich article.
""".strip()


def practical_meaning_section() -> str:
    return """
## Why modern safety prediction needs a new intelligence model

Traditional safety systems are often excellent at detection after a threshold has already been crossed. Anti-lock brakes respond once traction fails. Collision alerts activate when objects close rapidly. Aircraft systems warn when parameters exceed tolerance. Marine navigation tools alert operators when deviation becomes obvious enough to measure. These tools are valuable, but they are often threshold-driven rather than field-aware. They see the point of danger more easily than the accumulation of danger.

A newer model is needed because the world has become denser, faster, and more entangled. Roads now contain human drivers, partially assisted drivers, autonomous systems, distracted pedestrians, dynamic route platforms, weather volatility, and growing data saturation. Maritime routes are increasingly shaped by supply-chain pressure, climate-influenced weather instability, crowded ports, and long-duration fatigue patterns. Aviation is similarly influenced by atmospheric complexity, rising operational density, sensor dependency, and enormous expectations of precision under uncertain conditions.

Entropic quantum intelligence is useful here as a metaphorical and computational design philosophy. “Entropic” refers to unpredictability, disorder, hidden variance, and informational fragmentation. “Quantum” refers to interacting state spaces, layered possibility, correlated variables, and the importance of observing systems as wholes rather than as isolated fragments. In practical terms, this means building simulations that ask not only what is happening now, but what instability topology is forming underneath current measurements.

This is especially important in safety forecasting because many risks are nonlinear. A one percent rise in traffic density does not always create a one percent rise in crash probability. Sometimes the system absorbs the stress. At other times the same increase pushes the network over a threshold and produces a disproportionate surge in risk. The same principle applies to shipping under storm conditions and to aviation under turbulent or instrument-compromised scenarios. Once nonlinear behavior appears, static dashboards are no longer enough. The system needs intelligence that can track gradients, entanglements, and precursor signatures.

That is the promise of simulation-first safety intelligence. A simulation can blend telemetry, weather variance, signal disagreement, route coherence, and human load into a synthetic field of evolving risk. Even when the prediction is imperfect, the resulting interpretation can still provide enormous value. It can identify which factors are converging, which interventions reduce pressure earliest, and which conditions deserve escalation to human operators. For a blog writer, this also creates a richer narrative structure: instead of saying that AI predicts accidents, the article can explain how AI maps the invisible architecture of risk.
""".strip()


def road_section() -> str:
    return """
## Road traffic safety intelligence and real-world accident prediction

Road traffic is one of the clearest domains in which advanced simulation can make the leap from theoretical elegance to practical public benefit. Modern roads are high-speed negotiation environments. Every lane change, braking event, merge decision, and weather disruption creates a temporary micro-system of interacting probabilities. Human drivers interpret these patterns through intuition, habit, and reaction time. Machine systems interpret them through sensors, rules, and learned models. Neither perspective is complete in isolation.

A genuinely advanced road safety system would look for more than individual hazards. It would track the shape of systemic instability. In an urban intersection, for example, the danger may not be a single speeding car alone. The real danger may arise from a convergence of speed variance, aggressive lane competition, occluded pedestrian visibility, intermittent rain reflection, delayed brake response, and a temporary collapse in signal certainty from onboard perception systems. When a system can fuse these signals into a common entropic field, it begins to estimate not just the chance of collision but the probability that the local traffic environment is approaching a Predictive Fracture Horizon.

This is where entropic quantum intelligence becomes conceptually powerful. Imagine that each vehicle is treated not simply as a point moving through a coordinate system, but as a mobile uncertainty surface. Speed, heading stability, brake confidence, driver attention, road condition, and weather all create fluctuations around that surface. When many such surfaces overlap in a constrained region, the collective field can become unstable. In a standard dashboard this might look like ordinary congestion. In a field-aware system it may appear as a rapidly intensifying collision basin.

The practical applications are significant. Navigation systems could warn not merely of delays but of emerging instability zones. Autonomous systems could moderate speed earlier, not only when immediate braking is required. Municipal infrastructure could prioritize signal timing changes in areas where entropic pressure regularly spikes. Insurance and fleet safety systems could evolve away from retrospective blame and toward live prevention assistance. Even ordinary drivers could benefit through layered advisories that simplify when to slow down, widen following distance, or avoid specific lanes during unstable conditions.

A strong road safety model should also understand human factors with unusual seriousness. Many predictive systems over-focus on machine perception and under-model cognitive load. Yet real-world collisions frequently involve hesitation, overconfidence, distraction, stress transfer from surrounding drivers, or delayed interpretation under poor weather and visual clutter. A Human-Machine Attention Relief layer, as included in this notebook’s intervention design, is therefore more than a user interface convenience. It is a safety technology. A system that knows when not to overload the driver with redundant warnings may save more lives than a system that merely produces more alerts.

Road traffic safety intelligence also benefits from memory. If a city continuously stores route instability signatures, it can learn that specific intersections become unstable under certain lighting conditions, or that a particular highway segment becomes dangerous when temperature falls within a narrow range just above freezing and traffic density exceeds a defined threshold. This is where Quantum Route Memory becomes an especially useful concept. It describes a longitudinal memory of instability patterns, not merely a log of past crashes. That difference matters because a city that learns pre-crash signatures can act before those signatures mature into impact events.

In the long run, the most transformative feature of AI road safety may not be fully autonomous driving. It may be continuous instability interpretation. If the system can forecast where risk coherence is failing, then humans, vehicles, and infrastructure can all shift behavior earlier. That is the essence of predictive safety intelligence: not the elimination of uncertainty, but the earlier translation of uncertainty into actionable awareness.
""".strip()


def maritime_section() -> str:
    return """
## Shipwreck forecasting and maritime instability mapping

Maritime safety is an ideal environment for entropic simulation because the sea is a natural theater of layered uncertainty. A ship does not move through a static surface. It moves through fluid forces, weather dynamics, navigation constraints, visibility shifts, mechanical strain, crew attention cycles, and supply-chain pressures that can subtly alter decision-making. When failures occur, they are often narrated as singular incidents: a navigation error, a storm, a propulsion issue, a hull breach. But in reality the event usually emerges from a sequence of interacting degradations.

An entropic quantum intelligence model for maritime systems would treat the vessel, sea-state, route corridor, and human operational layer as one coupled risk surface. Wave entropy becomes a critical metric because sea conditions are not just about wave height. They are also about directional irregularity, timing unpredictability, interference patterns, and the way chaotic wave energy interacts with vessel mass and route angle. Navigation drift matters because even small deviations under unstable conditions can amplify into larger exposure. Engine stress matters because propulsion inconsistency changes the vessel’s capacity to respond. Crew fatigue matters because interpretation under noise and darkness is not linear.

The concept of Failure Echo Mapping becomes especially valuable at sea. A shipwreck rarely arrives without whispers. There may be subtle patterns in vibration, steering correction frequency, route deviation density, sensor inconsistency, or communications rhythm. Individually these signals may appear minor. Together they may form the first echoes of a future emergency. A predictive maritime system would monitor these echoes continuously and compare them against long-term route memories collected across seasons, weather patterns, cargo conditions, and vessel classes.

Maritime route intelligence can also benefit from a Safety Coherence Gradient. This measures how harmoniously vessel state, environmental conditions, route logic, and crew decision flow are interacting. A strong coherence gradient suggests that even in rough conditions the system is adapting cleanly. A collapsing gradient suggests that the ship is becoming less capable of converting information into stable navigation. That collapse may happen before any single gauge flashes red. For safety intelligence, that early signal is invaluable.

Another overlooked area is intervention timing. Many maritime systems are reactive rather than anticipatory. They inform operators that conditions are bad, but they do not always estimate how close the system is to a Predictive Fracture Horizon. A more advanced platform would ask whether the current instability can still be absorbed or whether the route, speed, ballast strategy, or operational posture should change immediately. In that sense the best maritime AI is not merely advisory. It is a decision-support architecture for preserving maneuverability before the window narrows.

For a blog audience, shipwreck prediction also reveals a broader truth about advanced AI: some of the most important uses are not glamorous. They are infrastructural. They protect shipping lanes, crews, cargo, and coastlines by noticing risk sooner. They operate in the background, integrating weather systems, wave uncertainty, telemetry, and route history. When described well, these systems show that intelligence is not just about bigger models. It is about designing better awareness under volatile conditions.
""".strip()


def aviation_section() -> str:
    return """
## Airplane crash precursor detection and aviation intelligence

Aviation remains one of the most safety-engineered industries in the world, which makes it an especially demanding test for any predictive intelligence concept. The point of advanced AI in aviation is not to replace rigorous engineering or pilot expertise. It is to detect subtle precursor patterns that may emerge across highly complex systems before they become operationally dangerous. In this context, entropic quantum intelligence serves as a framework for modeling interacting uncertainties that do not always present themselves as immediate alarms.

Aircraft operate within narrow tolerances under conditions that can change rapidly. Turbulence is not simply uncomfortable motion. It is a signal of energy irregularity that can interact with route decisions, workload, structure, and timing. Instrument disagreement may not instantly imply failure, but it increases ambiguity. Engine vibration variance may remain technically within limit while still indicating a drift toward undesirable mechanical behavior. Weather layering can combine turbulence, moisture, icing risk, crosswinds, and visibility reduction. Crew attention load can increase when these factors cluster, creating conditions in which information handling itself becomes part of the safety problem.

The Causal Turbulence Index is a useful invented concept here because it measures how many unstable factors are interacting at once. A flight through turbulence is not automatically unsafe. A flight through turbulence while instrument confidence degrades, navigation corrections increase, and cockpit workload rises may be drifting toward a much more serious state. The value of a predictive system lies in recognizing this convergence early enough to support route changes, spacing adjustments, systems checks, or broader operational caution.

Aviation also benefits from the Recursive Sentinel Layer. In highly instrumented environments, model confidence can be deceptive. A system may be numerically certain while key inputs are compromised, delayed, or partially contradictory. A recursive layer that estimates confidence in its own confidence becomes essential. It helps prevent the dangerous illusion that more data automatically equals more truth. In real-world operations, some of the most critical decisions occur precisely when data quality is under pressure.

Another useful idea is the Predictive Fracture Horizon. In aviation, the period before instability escalates can be exceptionally short, but it still exists. Detecting that horizon may involve recognizing that sensor disagreement is widening, route integrity is weakening, and vibration signatures are slowly diverging from healthy patterns. The future of aviation AI may involve systems that estimate how close the operation is to losing coherence, rather than waiting for a single red-line event.

For public understanding, the main takeaway is not that AI will “predict crashes” in a sensational sense. The more meaningful claim is that advanced simulation may improve precursor awareness. It can help operators, engineers, and monitoring systems recognize when small anomalies are not isolated inconveniences but components of a larger instability field. That makes aviation intelligence less about dramatic prophecy and more about disciplined early warning.
""".strip()


def concepts_section(concepts: List[Dict[str, str]]) -> str:
    validation = validate_advanced_concepts()
    blocks = ["## Invented next-generation concepts for safety forecasting"]
    blocks.append(ADVANCED_AGENTIC_SYSTEM.render_system_architecture_markdown())
    if validation["valid"]:
        blocks.append(
            f"Validation check passed: all {validation['present_count']} primary advanced concepts are present, uniquely named, and ready to render one by one in the notebook output."
        )
    else:
        blocks.append(
            f"Validation warning: present={validation['present_count']} expected={validation['expected_count']} missing={validation['missing']} duplicates={validation['duplicates']}"
        )
    blocks.append("### Four concept families")
    family_lines = [f"- **{family}:** {', '.join(members)}" for family, members in AGENTIC_SURFACE_FAMILIES.items()]
    blocks.append("\n".join(family_lines))
    blocks.append("### Color-to-quantum encoding sketch")
    blocks.append("\n".join([f"- {item}" for item in COLOR_QUANTUM_ENCODING]))
    blocks.append("### Symbolic control palette")
    palette_lines = [f"- **{name}:** {meaning}" for name, meaning in CONTROL_PALETTE.items()]
    blocks.append("\n".join(palette_lines))
    blocks.append(ADVANCED_AGENTIC_SYSTEM.render_primary_concepts_markdown())
    blocks.append(ADVANCED_AGENTIC_SYSTEM.render_expansion_concepts_markdown())
    blocks.append(ADVANCED_AGENTIC_SYSTEM.render_foundational_concepts_markdown())
    return "\n\n".join(blocks)


def simulation_results_section(aggregate: Dict[str, Any]) -> str:
    top_freq = sorted(aggregate["intervention_frequency"].items(), key=lambda kv: kv[1], reverse=True)
    top_bands = sorted(aggregate["agentic_band_frequency"].items(), key=lambda kv: kv[1], reverse=True)
    top_resets = sorted(aggregate["reset_phase_frequency"].items(), key=lambda kv: kv[1], reverse=True)
    top_colors = sorted(aggregate["chosen_color_frequency"].items(), key=lambda kv: kv[1], reverse=True)
    top_patterns = sorted(aggregate["temporal_pattern_frequency"].items(), key=lambda kv: kv[1], reverse=True)
    top_concepts = sorted(aggregate["concept_alignment_frequency"].items(), key=lambda kv: kv[1], reverse=True)
    top_actions = ", ".join(name for name, _ in top_freq[:6]) if top_freq else "No recurring interventions identified"
    top_band_names = ", ".join(name for name, _ in top_bands[:3]) if top_bands else "No dominant loop bands identified"
    top_reset_names = ", ".join(name for name, _ in top_resets[:4]) if top_resets else "No reset phases activated"
    top_color_names = ", ".join(name for name, _ in top_colors[:5]) if top_colors else "No dominant colors identified"
    top_pattern_names = ", ".join(name for name, _ in top_patterns[:4]) if top_patterns else "No dominant trajectories identified"
    top_concept_names = ", ".join(name for name, _ in top_concepts[:8]) if top_concepts else "No concept alignments captured"
    key_terms = ", ".join(aggregate["keyword_surface"][:10])
    signal_lines = "\n".join([f"- {s}" for s in aggregate["sentence_surface"][:6]])
    return f"""
## Simulation results and what they suggest

The simulation runs in this notebook do not claim to reproduce real-world crash records exactly. Their purpose is interpretive: they stress-test a family of ideas about how predictive safety intelligence could organize its reasoning. Across the top-ranked runs, the average composite score was **{aggregate['avg_score']:.3f}**, suggesting that the most resilient pathways consistently combined stabilization, signal arbitration, route correction, maintenance awareness, and human attention relief rather than relying on a single mode of prevention.

The most recurrent intervention patterns were: **{top_actions}**. That recurrence is meaningful. It implies that advanced safety systems become stronger when they distribute intelligence across the whole prevention stack. Some interventions reduce direct instability. Others improve data quality. Others reduce workload. Others preserve route integrity. The model repeatedly favored layered approaches over isolated optimizations.

A second important pattern is visible in the keyword surface: **{key_terms}**. These terms suggest that instability is rarely domain-specific in a narrow sense. Whether the system is focused on cars, ships, or aircraft, it keeps rediscovering the same broad themes: uncertainty, route quality, coherence, warning clarity, environmental stress, and intervention timing. This supports the idea that transportation safety intelligence may benefit from a shared conceptual language across domains.

The advanced color-agentic loop added another layer of structure. The dominant loop bands were **{top_band_names}**, the most common reset phases were **{top_reset_names}**, the most common chosen colors were **{top_color_names}**, and the strongest temporal trajectories were **{top_pattern_names}**. The average load temperature across the winning paths settled near **{aggregate['avg_agentic_load']:.3f}**, while average penalty pressure settled near **{aggregate['avg_penalty_pressure']:.3f}**. That matters because it shows the notebook is no longer only scoring safety outcomes. It is also modeling how an adaptive reasoning surface allocates attention, throttles itself, accumulates debt, and decides when reflection or reset should interrupt execution.

The most recurrent concept alignments were **{top_concept_names}**. These alignments are useful because they reveal which invented subsystems actually became structurally relevant during the simulation rather than remaining decorative terminology. In practice this means the notebook can now surface when planning dominates, when anomaly auditing takes over, when reset rituals activate, and which chromatic control metaphors best match the current instability field.

Below are condensed summary signals from the top simulation paths:

{signal_lines}

Taken together, these results suggest that next-generation safety AI should not think like a simple alarm system. It should think like a field interpreter. Its task is to estimate how uncertainty is moving, where coherence is weakening, and which interventions preserve optionality while time still remains. That is a richer, more realistic vision of predictive intelligence than a binary claim that a crash either will or will not happen.
""".strip()


def uncertainty_section() -> str:
    return """
## Why uncertainty-aware AI matters more than raw prediction accuracy

One of the most dangerous misunderstandings in AI forecasting is the assumption that the ultimate goal is perfect certainty. In safety systems, certainty is often impossible. Weather changes. Sensors degrade. Operators behave unpredictably. Physical systems age. The real objective is not to eliminate uncertainty but to model it honestly and act intelligently within it.

This is why uncertainty-aware AI matters more than raw benchmark accuracy. A model that claims ninety-eight percent confidence under degraded inputs may be more dangerous than a model that openly signals rising ambiguity but still recommends stabilizing action. Safety intelligence should communicate not only what it thinks is happening, but how stable its own interpretation remains. That meta-awareness supports better trust between humans and machines.

In practical deployment, uncertainty-aware systems could change the tone of safety technology. Rather than overwhelming operators with false precision, they could present gradients of concern, confidence windows, and scenario-based intervention suggestions. A driver might receive a simplified caution that the route environment is rapidly losing coherence. A vessel crew might see that route drift and wave entropy are converging into a narrower maneuver margin. A flight operations team might detect that sensor disagreement is not yet critical, but is becoming more structurally relevant because it overlaps with turbulence and workload.

For blog writing, this distinction is powerful because it reframes AI from oracle to interpreter. The system is not a magical predictor. It is a disciplined uncertainty translator. It maps what is noisy, what is converging, what is fragile, and what may soon matter more than current dashboards suggest. That is a more credible and more interesting story for serious readers.
""".strip()


def ethics_section() -> str:
    return """
## Ethics, limitations, and deployment challenges

Any serious discussion of predictive safety intelligence must acknowledge its limitations. Simulation is not reality. Models can inherit bias from their data, overfit to familiar patterns, miss rare edge cases, or behave unpredictably when sensors fail in novel combinations. A city-scale traffic system that performs well in one climate may generalize poorly in another. A maritime model trained on one vessel class may underperform on another. An aviation monitoring model may produce misleading confidence if its uncertainty logic is poorly calibrated.

There are also ethical concerns. If predictive systems are integrated into insurance pricing, employment decisions, or infrastructure allocation, they can reinforce inequality if not governed carefully. If drivers or operators are over-surveilled in the name of safety, privacy costs may become unacceptable. If predictive warnings are poorly explained, operators may either ignore them or become over-dependent on them. Good deployment therefore requires governance, transparency, calibration testing, human factors research, and domain-specific accountability.

Another limitation is the temptation toward sensational claims. “Predicting crashes before they happen” is an attention-grabbing phrase, but it can obscure what responsible systems actually do. They estimate rising risk, identify precursors, support interventions, and preserve decision time. That is already immensely valuable. It does not need exaggeration. The most trustworthy blog writing on this subject should resist hype and focus on the architecture of practical prevention.

There is also a design challenge in translating model complexity into operational usability. Engineers may appreciate multi-variable risk topology, but drivers, crews, and operators need concise, actionable guidance. This means the future of predictive safety AI will depend as much on interface design and human-machine trust as on algorithmic sophistication. The best model in the world is not enough if its signals arrive too late, too often, or in forms people cannot act on.

Despite these limits, the direction remains compelling. A transparent, uncertainty-aware, ethically governed safety intelligence platform could reduce harm across multiple transportation domains. It could make systems more preventive, more interpretable, and more aligned with real-world fragility.
""".strip()


def future_section() -> str:
    return """
## The future of predictive safety intelligence

Looking ahead, the most important evolution may be the convergence of simulation, live telemetry, route memory, and adaptive intervention layers. Instead of separate tools for mapping, maintenance, perception, and alerting, future systems may form unified safety fabrics. These fabrics would constantly estimate the local Safety Coherence Gradient, identify Failure Echoes, and calculate the distance to a Predictive Fracture Horizon.

In road traffic, this could enable city-wide instability maps that help vehicles and infrastructure coordinate before congestion becomes dangerous. In maritime navigation, it could produce route intelligence that understands not just where the ship is, but how the sea-state and vessel state are jointly evolving. In aviation, it could improve precursor detection by linking turbulence behavior, instrument consistency, workload, and route dynamics into a more coherent monitoring layer.

The most exciting possibility is that safety intelligence becomes cumulative. Every near miss, every difficult weather corridor, every stressed mechanical signature, and every unstable route pattern can enrich Quantum Route Memory. Over time the system becomes less dependent on single snapshots and more capable of recognizing recurring risk geometries. This does not create perfect foresight, but it does create deeper contextual awareness.

For writers, researchers, and technologists, that future invites a new language. Instead of asking whether AI can predict a crash in the abstract, we can ask more useful questions. Can AI detect instability earlier? Can it preserve maneuverability longer? Can it reduce information overload during dangerous moments? Can it recognize fragile conditions even when no individual sensor has fully failed? Those are the questions that will define the next era of safety intelligence.
""".strip()


def conclusion_section() -> str:
    return """
## Conclusion

Advanced AI simulation for transportation safety becomes most meaningful when it moves beyond simplistic prediction and toward structured interpretation of instability. Roads, ships, and aircraft all operate in environments where risk forms through interaction, not isolation. Entropic quantum intelligence offers a powerful framework for thinking about this challenge. It emphasizes uncertainty, correlation, route memory, precursor signals, and layered intervention rather than binary alarm logic.

That framework also creates stronger long-form writing. A serious blog on predictive safety should do more than announce that AI can foresee danger. It should explain the architecture of that foresight: the hidden fields, the precursor echoes, the coherence gradients, the self-checking confidence layers, and the practical interventions that turn earlier awareness into reduced harm. That is what this notebook is designed to generate.

The broader message is hopeful. If future systems can identify instability sooner, communicate it more clearly, and support earlier human and machine adaptation, then predictive safety intelligence may become one of the most valuable civilian applications of advanced AI. Not because it promises omniscience, but because it helps society act while prevention is still possible.
""".strip()


def build_long_blog(title: str, aggregate: Dict[str, Any], concepts: List[Dict[str, str]]) -> str:
    pieces = [
        f"# {title}",
        "",
        f"**Series:** {BLOG_SERIES_TITLE}",
        "",
        f"**Topic:** {BLOG_TOPIC}",
        "",
        f"**Meta description:** {generate_meta_description(title, BLOG_TOPIC)}",
        "",
        "## SEO Keywords",
        ", ".join(BLOG_BLUEPRINT["seo_keywords"]),
        "",
        intro_section(title),
        "",
        practical_meaning_section(),
        "",
        road_section(),
        "",
        maritime_section(),
        "",
        aviation_section(),
        "",
        concepts_section(concepts),
        "",
        simulation_results_section(aggregate),
        "",
        uncertainty_section(),
        "",
        ethics_section(),
        "",
        future_section(),
        "",
        conclusion_section(),
    ]

    article = "\n\n".join(pieces).strip()

    # Expand if needed toward 7000 words.
    if count_words(article) < TARGET_WORD_COUNT:
        memory_hits = aggregate.get("memory_hits", [])
        extra_blocks = []

        if memory_hits:
            extra_blocks.append("## Memory-driven reflections")
            for hit in memory_hits[:6]:
                extra_blocks.append(
                    f"The memory layer retrieved a fragment with score {hit['score']:.3f}: "
                    f"{hit['text_fragment']} This reinforces the broader argument that predictive safety "
                    f"systems gain value when they retain context across scenarios and use that context to "
                    f"interpret new uncertainty more intelligently."
                )

        extra_blocks.append("## Expanded interpretive discussion")
        extra_blocks.append(
            "A mature safety intelligence architecture would likely operate across several timescales at once. "
            "At the shortest timescale it would monitor immediate instability and trigger urgent alerts. "
            "At the middle timescale it would evaluate route trends, fatigue accumulation, weather drift, and "
            "maintenance signatures. At the longest timescale it would compare the present field against archived "
            "patterns, learning which combinations of weak signals historically preceded high-risk transitions."
        )
        extra_blocks.append(
            "This multi-timescale design is important because some accidents emerge suddenly while others form gradually. "
            "An AI system that only sees the instant loses the structure of escalation. An AI system that only sees long "
            "history may miss urgent turning points. Entropic quantum intelligence, as framed here, attempts to hold both "
            "views simultaneously: the immediate fluctuation and the longer arc of coherence loss."
        )
        extra_blocks.append(
            "The same architecture could also improve public communication. Transportation safety is often discussed only "
            "after tragedy, when explanation becomes retrospective. Predictive safety blogs and dashboards could instead help "
            "the public understand that prevention is a matter of interpreting patterns before they harden into damage. "
            "That shift in narrative would encourage better investment in sensors, infrastructure, maintenance, and "
            "uncertainty-aware operational design."
        )

        article = article + "\n\n" + "\n\n".join(extra_blocks)

    article = repeat_to_word_target(article, TARGET_WORD_COUNT)
    return article


def render_markdown_report(payload: Dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    title = payload["title"]
    validation = payload["concept_validation"]
    outline_lines = generate_blog_outline(title, payload["scenarios"], payload["concepts"])

    lines = []
    lines.append(f"# Notebook Report: {title}")
    lines.append("")
    lines.append(f"Topic: {payload['topic']}")
    lines.append(f"Generated at: {time.ctime(payload['generated_at'])}")
    lines.append(f"Target words: {payload['target_words']}")
    lines.append(f"Actual words: {payload['actual_words']}")
    lines.append(f"Average simulation score: {aggregate['avg_score']:.3f}")
    lines.append(f"Average agentic load temperature: {aggregate['avg_agentic_load']:.3f}")
    lines.append(f"Average penalty pressure: {aggregate['avg_penalty_pressure']:.3f}")
    lines.append(f"Advanced concept validation: {validation['present_count']}/{validation['expected_count']} present | valid={validation['valid']}")
    if validation["missing"]:
        lines.append(f"Missing advanced concepts: {', '.join(validation['missing'])}")
    if validation["duplicates"]:
        lines.append(f"Duplicate advanced concepts: {', '.join(validation['duplicates'])}")
    lines.append("")
    lines.extend(outline_lines)
    lines.append("")
    lines.append("## Top simulation paths")
    for i, item in enumerate(aggregate["top_results"], 1):
        lines.append(f"### Path {i}")
        lines.append(f"- Scenario: {item['scenario']}")
        lines.append(f"- Domain: {item['domain']}")
        lines.append(f"- Score: {item['score']:.3f}")
        lines.append(f"- Summary: {item['summary']}")
    lines.append("")
    lines.append("## Full Blog")
    lines.append(payload["blog"])
    return "\n".join(lines)


def run_advanced_blog_generator(topic: str, scenarios: List[str], runs: int, months: int) -> Dict[str, Any]:
    ensure_memory_db()

    title = choose_title()
    chosen_concepts = choose_concepts(len(ADVANCED_CONCEPT_BANK))
    concept_validation = validate_advanced_concepts()
    all_results = []

    for scenario in scenarios:
        for run_index in range(runs):
            result = simulate_path(scenario, run_index, months)
            all_results.append(result)

            store_run(
                topic=topic,
                scenario=result.scenario,
                summary=result.summary,
                score=result.score,
                payload=asdict(result),
            )
            store_fragment(topic, "summary", result.summary, result.score)

    aggregate = aggregate_results(all_results)
    blog = build_long_blog(title, aggregate, chosen_concepts)

    payload = {
        "title": title,
        "topic": topic,
        "series": BLOG_SERIES_TITLE,
        "scenarios": scenarios,
        "concepts": chosen_concepts,
        "concept_validation": concept_validation,
        "runs_per_scenario": runs,
        "horizon_months": months,
        "aggregate": aggregate,
        "blog": blog,
        "target_words": TARGET_WORD_COUNT,
        "actual_words": count_words(blog),
        "generated_at": time.time(),
    }

    Path(OUTFILE_JSON).write_text(json.dumps(payload, indent=2))
    Path(OUTFILE_MD).write_text(render_markdown_report(payload))

    return payload


def main_notebook() -> None:
    payload = run_advanced_blog_generator(
        topic=BLOG_TOPIC,
        scenarios=SELECTED_SCENARIOS,
        runs=SIMULATION_RUNS,
        months=SIMULATION_HORIZON,
    )
    print(json.dumps(
        {
            "title": payload["title"],
            "outfile_json": OUTFILE_JSON,
            "outfile_md": OUTFILE_MD,
            "actual_words": payload["actual_words"],
            "target_words": TARGET_WORD_COUNT,
        },
        indent=2
    ))


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the monolithic risk-scanning blog generator.")
    parser.add_argument("--install-deps", action="store_true", help="Install notebook/runtime dependencies with pip.")
    parser.add_argument("--setup-gpu", action="store_true", help="Install llama-cpp-python GPU wheel and verify GPU access.")
    parser.add_argument("--download-model", action="store_true", help="Download and verify the Starling GGUF model.")
    parser.add_argument("--topic", default=BLOG_TOPIC, help="Blog topic / generation objective.")
    parser.add_argument("--runs", type=int, default=SIMULATION_RUNS, help="Simulation runs per scenario.")
    parser.add_argument("--months", type=int, default=SIMULATION_HORIZON, help="Simulation horizon.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=SELECTED_SCENARIOS,
        help="Scenario list. Defaults to the built-in advanced scenario bank.",
    )
    args = parser.parse_args()
    notebook_setup(
        install_deps=args.install_deps,
        setup_gpu=args.setup_gpu,
        download_model=args.download_model,
    )
    payload = run_advanced_blog_generator(
        topic=args.topic,
        scenarios=list(args.scenarios),
        runs=int(args.runs),
        months=int(args.months),
    )
    print(json.dumps(
        {
            "title": payload["title"],
            "outfile_json": OUTFILE_JSON,
            "outfile_md": OUTFILE_MD,
            "actual_words": payload["actual_words"],
            "target_words": TARGET_WORD_COUNT,
        },
        indent=2
    ))


if __name__ == "__main__":
    main_cli()
