from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as np


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
