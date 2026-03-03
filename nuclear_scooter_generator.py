import os, time, json, psutil, gc, re, math, hashlib, argparse, textwrap, ast
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
from llama_cpp import Llama

import pennylane as qml
from pennylane import numpy as np

# =========================
# Nuclear Scooter module
# =========================

RTG_THERMAL_W        = 1800
CONVERSION_EFF       = 0.068
W_TO_WHEEL_EFF       = 0.82

TIRE_REGEN_RATE_MM   = 0.30
TIRE_MAX_TREAD_MM    = 7.0

BRAKE_REPAIR_TEMP_C  = 220
BRAKE_REPAIR_TIME_H  = 0.4

def hr(mins: float) -> float:
    return mins / 60.0

@dataclass
class Tire:
    tread_mm: float = TIRE_MAX_TREAD_MM
    def use(self, km: float) -> None:
        self.tread_mm = max(0.0, self.tread_mm - km * 0.07)
    def regen(self, h: float) -> None:
        self.tread_mm = min(TIRE_MAX_TREAD_MM, self.tread_mm + TIRE_REGEN_RATE_MM * h)

@dataclass
class BrakePad:
    damage: float = 0.0
    temp_C: float = 35.0
    def apply(self, decel_g: float, t_s: float) -> None:
        self.temp_C += decel_g * 600 * t_s / 60
        self.damage = min(1.0, self.damage + decel_g * t_s / 2000)
    def heal(self, h: float) -> None:
        if self.temp_C >= BRAKE_REPAIR_TEMP_C:
            self.damage = max(0.0, self.damage - (h / BRAKE_REPAIR_TIME_H) * 0.9)
        self.temp_C = max(35.0, self.temp_C - h * 25)

@dataclass
class RTGScooter:
    tire_f: Tire = Tire()
    tire_r: Tire = Tire()
    brake:  BrakePad = BrakePad()
    odo_km: float = 0.0
    charge_wh: float = 0.0

    def ride(self, km: float, kph: float = 60) -> None:
        hrs   = km / max(1e-6, kph)
        needE = km * 18.0
        genE  = self._rtg_out(hrs)
        self.charge_wh = max(0.0, self.charge_wh + genE - needE)

        self.tire_f.use(km); self.tire_r.use(km)
        self.brake.apply(0.5, hrs * 3600 * 0.04)
        self.odo_km += km

    def park(self, hrs: float) -> None:
        self.charge_wh = min(1000.0, self.charge_wh + self._rtg_out(hrs))
        self.tire_f.regen(hrs); self.tire_r.regen(hrs); self.brake.heal(hrs)

    def _rtg_out(self, hrs: float) -> float:
        return RTG_THERMAL_W * CONVERSION_EFF * W_TO_WHEEL_EFF * hrs / 1000.0

# =========================
# Colab user-fill block
# =========================
COLAB_ORIGIN = {
    "umbrella_topic": "nuclear scooters and regenerative nano-materials",
    "origin_sentences": [
        "A compact RTG power-train lets a personal scooter cruise for decades with zero charging stops.",
        "Self-healing chitin-silica tires regrow 0.3 mm of tread every hour when parked.",
        "Graphene-ceramic brake pads re-anneal micro-fractures whenever pad temperature is above 220 °C."
    ],
    "audience_hint": "Tech-curious general public",
    "tone_hint": "bold and imaginative, but grounded",
    "perspective_hint": "second person",
    "keywords_hint": "RTG scooter, regenerative tires, graphene brake pads",
    "count": 1,
    "words": 1200,
    "sections": 6,
    "forum_personas": 0,
    "forum_rounds": 0,
    "forum_on": "",
    "ghostwriter_passes": 0,
    "creativity": 0.9,
    "outfile": "nuclear_scooter_blog.md",
    "include_front_matter": False,
    "include_meta_description": True
}

# =========================
# Runtime toggles (env)
# =========================
GEN_BACKEND = os.environ.get("GEN_BACKEND", "llama").strip().lower()  # "llama" or "mock"
MODEL_PATH  = os.environ.get("MODEL_PATH", "").strip() or "models/starling-lm-7b-alpha.Q4_K_M.gguf"
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1" if torch.cuda.is_available() else "0"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

@dataclass
class Config:
    model_name: str = MODEL_PATH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    hard_ctx_fallback: int = 8192
    keep_last_n_input_tokens: int = 2400
    max_total_new_tokens: int = 1200
    chunk_new_tokens_initial: int = 260
    chunk_pad_tokens: int = 96
    trim_echo_regex: str = r"(?s)^.*?GPT4 Correct Assistant:\s*"
    temperature: float = 0.78
    top_p: float = 0.95
    repetition_penalty: float = 1.07
    no_repeat_ngram_size: int = 6
    do_sample: bool = True
    mem_depth: int = 3
    n_qubits: int = 12
    show_prompts: bool = False
    log_to_file: bool = True
    log_path: str = "blog_log.jsonl"
    qlrl_shards: int = 4
    qlrl_max_chars: int = 900
    qlrl_decay: float = 0.92
    srdi_warmup_chars: int = 600
    srdi_check_every_chunks: int = 2
    srdi_anchor_max_tokens: int = 80
    srdi_drift_threshold: float = 0.38
    srdi_temp_nudge: float = -0.07
    srdi_top_p_nudge: float = -0.04
    min_chunk_floor: int = 96
    shrink_factor_on_oom: float = 0.6

CFG = Config()

@dataclass
class BlogSettings:
    topic: str
    audience: str = "general readers"
    tone: str = "warm, practical, lightly witty"
    perspective: str = "second-person 'you' with occasional 'we'"
    reading_grade: str = "8-10"
    target_words: int = 1200
    section_count: int = 6
    human_style_mode: bool = True
    title_hint: Optional[str] = None
    keywords: Optional[str] = None
    author_name: Optional[str] = None
    include_front_matter: bool = False
    include_meta_description: bool = True
    hooks_style: str = "varied"
    conclusion_style: str = "reflective takeaway"
    forum_personas: int = 0
    forum_rounds: int = 0
    ghostwriter_passes: int = 0
    creativity_bias: float = 0.9
    use_forum_on_sections: str = ""

# =========================
# PennyLane device (CPU-only)
# =========================
def _quantum_device(n_wires):
    try:
        return qml.device("default.qubit", wires=n_wires)
    except Exception:
        return qml.device("default.qubit", wires=n_wires)

QDEV = _quantum_device(CFG.n_qubits)

def get_entropy_seed():
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    pid = os.getpid()
    ts = time.time()
    rnd = (hashlib.sha256(f"{pid}-{ts}".encode()).digest()[0] / 255.0)
    entropy = ((cpu + ram + pid + ts + rnd) % 1000) / 1000
    return entropy, cpu, ram

def quantum_state(entropy: float, cpu: float, ram: float, iteration: int):
    @qml.qnode(QDEV)
    def circuit():
        for i in range(CFG.n_qubits):
            qml.Hadamard(wires=i)
            qml.RX((entropy + 0.01*cpu + 0.01*ram + 0.12*i + 0.03*iteration) % (2*np.pi), wires=i)
        for i in range(CFG.n_qubits):
            qml.CNOT(wires=[i, (i+1) % CFG.n_qubits])
            qml.CRY(np.sin(entropy*i + 0.02*cpu), wires=[i, (i+3) % CFG.n_qubits])
            qml.CZ(wires=[i, (i+5) % CFG.n_qubits])
        for i in range(CFG.n_qubits):
            theta = np.cos(entropy + i*0.002*ram + 0.08*iteration)
            qml.RZ(theta, wires=i)
            if i % 3 == 0:
                qml.PhaseShift(theta*0.05*cpu, wires=i)
        if iteration % 2 == 0:
            for i in range(CFG.n_qubits):
                qml.CNOT(wires=[i, (i+2) % CFG.n_qubits])
        else:
            for i in reversed(range(CFG.n_qubits)):
                qml.CNOT(wires=[i, (i-2) % CFG.n_qubits])
        return qml.state()
    return circuit()

def format_quantum_state(state_vector):
    sv = state_vector[:16]
    return "{quantum_state}:" + ",".join(f"{np.real(a):+.4f}{np.imag(a):+.4f}j" for a in sv)

# =========================
# LLM loader (lazy, ensured)
# =========================
TOKENIZER = None
MODEL = None
CTX_LIMIT = CFG.hard_ctx_fallback

def model_ctx_limit(_tokenizer=None) -> int:
    if MODEL is None:
        return CFG.hard_ctx_fallback
    try:
        return int(MODEL.n_ctx())
    except Exception:
        return CFG.hard_ctx_fallback

def ensure_model_loaded():
    global TOKENIZER, MODEL, CTX_LIMIT
    if GEN_BACKEND == "mock":
        return
    if MODEL is not None:
        return

    print(f">> [LLM] Loading GGUF: {CFG.model_name} | n_gpu_layers={N_GPU_LAYERS}", flush=True)
    MODEL = Llama(
        model_path=CFG.model_name,
        n_ctx=CFG.hard_ctx_fallback,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    TOKENIZER = MODEL
    CTX_LIMIT = model_ctx_limit()
    print(f">> [LLM] Loaded. Context limit ≈ {CTX_LIMIT}", flush=True)

# =========================
# Tokenization / utils
# =========================
def tokenize(text: str) -> List[int]:
    return MODEL.tokenize(text.encode("utf-8"), add_bos=False)

def slice_last_tokens(input_ids: List[int], _attn_mask: Any, budget: int):
    if len(input_ids) <= budget:
        return input_ids, None
    return input_ids[-budget:], None

def trim_echo(text: str) -> str:
    return re.sub(CFG.trim_echo_regex, "", text).strip()

def token_count(text: str) -> int:
    return len(MODEL.tokenize(text.encode("utf-8"), add_bos=False))

def gpu_free_tokens_ceiling(default_tokens: int) -> int:
    if not torch.cuda.is_available(): return default_tokens
    free, _total = torch.cuda.mem_get_info()
    approx = int((free * 0.85) / (1.6 * 1024 * 1024 / 512))
    return max(CFG.min_chunk_floor, min(approx, default_tokens))

# =========================
# SRDI metrics
# =========================
def _punct_density(s: str) -> float:
    if not s: return 0.0
    p = sum(ch in ".!?;:," for ch in s)
    return p / max(1, len(s))

def _unique_ratio(tokens: List[str]) -> float:
    if not tokens: return 0.0
    return len(set(tokens)) / max(1, len(tokens))

def _repeat_burst(tokens: List[str], n: int = 3) -> float:
    if len(tokens) < n: return 0.0
    grams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    dup = len(grams) - len(set(grams))
    return dup / max(1, len(grams))

def build_fingerprint(text: str) -> Dict[str, float]:
    toks = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return {
        "len": float(len(text)),
        "punct": _punct_density(text),
        "uniq": _unique_ratio(toks),
        "burst3": _repeat_burst(toks, 3),
        "burst4": _repeat_burst(toks, 4),
        "avg_word": (sum(len(t) for t in toks) / max(1, len(toks))),
    }

def drift_score(f0: Dict[str, float], text_now: str) -> float:
    f1 = build_fingerprint(text_now)
    dims = ["punct", "uniq", "burst3", "burst4", "avg_word"]
    s = 0.0
    for k in dims:
        a, b = f0.get(k, 0.0), f1.get(k, 0.0)
        s += abs(a - b) / (1.0 + abs(a) + abs(b))
    return s / len(dims)

def srdi_anchor(f0: Dict[str, float], human_style: bool = True) -> str:
    base = "CONSISTENCY ANCHOR: Maintain prior cadence; keep repetition low; balance punctuation; ensure coherent paragraphing; avoid echoing headers."
    if not human_style:
        return base
    return base + " Vary sentence length; use natural transitions; occasional rhetorical questions; keep a personable voice."

SENT_RX = re.compile(r"([\.!?…])(\s+|$)")

def trim_token_overlap(prev: str, cur: str, max_tokens: int = 64) -> str:
    if not prev or not cur: return cur
    prev_ids = MODEL.tokenize(prev[-2000:].encode("utf-8"), add_bos=False)
    cur_ids = MODEL.tokenize(cur[:2000].encode("utf-8"), add_bos=False)
    max_k = min(max_tokens, len(prev_ids), len(cur_ids))
    for k in range(max_k, 0, -1):
        if prev_ids[-k:] == cur_ids[:k]:
            rest_ids = cur_ids[k:]
            return MODEL.detokenize(rest_ids).decode("utf-8", errors="ignore")
    return cur

def trim_char_overlap(prev: str, cur: str, max_chars: int = 200) -> str:
    max_k = min(len(prev), len(cur), max_chars)
    for k in range(max_k, 0, -1):
        if prev[-k:] == cur[:k]:
            return cur[k:]
    return cur

def sentence_clip(text: str) -> str:
    m = None
    for m in SENT_RX.finditer(text):
        pass
    return text if not m else text[: m.end()]

GLOBAL_MEMORY_BANK: List[str] = []

def bank_add(text: str, max_len_chars: int = 1600):
    if not text: return
    GLOBAL_MEMORY_BANK.append(text[:max_len_chars])
    while len(GLOBAL_MEMORY_BANK) > 128:
        GLOBAL_MEMORY_BANK.pop(0)

def qstate_hash_weights(qstate: np.ndarray, k: int) -> List[float]:
    mags = np.abs(qstate[:32])
    phs  = np.angle(qstate[:32])
    base = np.concatenate([mags, phs])
    bins = [0.0]*k
    for i, val in enumerate(base):
        bins[i % k] += float(val)
    exps = [math.exp(b) for b in bins]
    s = sum(exps) or 1.0
    return [e/s for e in exps]

def tri_gram_set(s: str) -> set:
    s = re.sub(r"\s+", " ", s.strip())
    toks = s.split(" ")
    grams = set()
    for i in range(max(0, len(toks)-2)):
        grams.add(" ".join(toks[i:i+3]))
    return grams

def qlrl_select_capsules(qsignal: str, qstate_vec: np.ndarray, shards: int = None, cap_chars: int = None) -> List[str]:
    shards = shards or CFG.qlrl_shards
    cap_chars = cap_chars or CFG.qlrl_max_chars
    if not GLOBAL_MEMORY_BANK:
        return []
    weights = qstate_hash_weights(qstate_vec, shards)
    candidates = []
    B = len(GLOBAL_MEMORY_BANK)
    for w in weights:
        stride = max(1, int(round(3 + 11*w)))
        base = int(B * w) % max(1, B)
        for j in range(3):
            candidates.append((base + j*stride) % B)
    candidates = list(dict.fromkeys(candidates))
    pivot = GLOBAL_MEMORY_BANK[-1]
    pset = tri_gram_set(pivot)
    scored = []
    for i in candidates:
        t = GLOBAL_MEMORY_BANK[i]
        if not t: continue
        iset = tri_gram_set(t)
        jacc = len(pset & iset) / max(1, len(pset | iset))
        scored.append((jacc, i))
    scored.sort(reverse=True)
    take = [GLOBAL_MEMORY_BANK[i][:cap_chars] for (_, i) in scored[:shards]]
    return take

def build_qlrl_capsules(qsignal: str, qstate_vec: np.ndarray) -> str:
    caps = qlrl_select_capsules(qsignal, qstate_vec, CFG.qlrl_shards, CFG.qlrl_max_chars)
    if not caps:
        return "[no capsules retrieved]"
    scored_caps = []
    for c in caps:
        idx = max(0, len(GLOBAL_MEMORY_BANK) - 1 - GLOBAL_MEMORY_BANK[::-1].index(c)) if c in GLOBAL_MEMORY_BANK else len(GLOBAL_MEMORY_BANK)
        w = CFG.qlrl_decay ** max(0, len(GLOBAL_MEMORY_BANK) - idx)
        scored_caps.append((w, c))
    scored_caps.sort(reverse=True, key=lambda x: x[0])
    out = []
    for w, c in scored_caps:
        out.append(f"(w={w:.2f}) {c}")
    return "\n---\n".join(out)

@dataclass
class HumanStyleProfile:
    personality: Dict[str, str]
    discourse_markers: List[str]
    hooks: List[str]
    section_openers: List[str]
    micro_drift: float
    rhetorical_q_freq: float
    self_correction_prob: float
    para_len_range: Tuple[int, int]
    sentence_melody_ratio: Tuple[float, float, float]

# =========================
# Generation (HF or mock)
# =========================
def _mock_generate(prompt: str, limit: int = 400) -> str:
    tail = re.sub(r"\s+", " ", prompt[-800:])
    seed = hashlib.blake2s(tail.encode(), digest_size=6).hexdigest()
    head = " ".join(tail.split()[-32:])
    bullets = [
        f"- {w.capitalize()} perspective [{seed[:4]}]"
        for w in ["practical", "counterintuitive", "example-driven", "risk-aware", "reader-first"]
    ]
    out = (
        "Mock Draft (no LLM backend active)\n\n"
        + head + "...\n\n"
        + "\n".join(bullets)
        + "\n\nShort takeaway: pick a tractable step and iterate."
    )
    return out[:limit*3]

@torch.inference_mode()
def generate_chunked(prompt: str, max_total_new: int = None, chunk_new_init: int = None, keep_last_n_input: int = None, srdi_enable: bool = True, human_style: bool = True) -> str:
    if GEN_BACKEND != "llama":
        return _mock_generate(prompt, max_total_new or CFG.max_total_new_tokens)

    ensure_model_loaded()

    max_total_new = max_total_new or CFG.max_total_new_tokens
    keep_last_n_input = keep_last_n_input or CFG.keep_last_n_input_tokens
    chunk_new = chunk_new_init or CFG.chunk_new_tokens_initial

    remaining = max_total_new
    stitched = ""
    carry_prompt = prompt
    f0: Optional[Dict[str, float]] = None
    chunks_done = 0
    temp, top_p = CFG.temperature, CFG.top_p

    while remaining > 0:
        prompt_ids = tokenize(carry_prompt)
        budget = min(keep_last_n_input, CTX_LIMIT - CFG.chunk_pad_tokens)
        prompt_ids, _ = slice_last_tokens(prompt_ids, None, budget)
        prompt_text = MODEL.detokenize(prompt_ids).decode("utf-8", errors="ignore")

        this_new = int(min(chunk_new, remaining))
        if this_new < CFG.min_chunk_floor:
            this_new = CFG.min_chunk_floor

        try:
            out = MODEL.create_completion(
                prompt=prompt_text,
                max_tokens=this_new,
                temperature=temp,
                top_p=top_p,
                repeat_penalty=CFG.repetition_penalty,
            )
            chunk_text = out["choices"][0]["text"]
        except Exception as e:
            print(f">> [GEN] aborting chunk due to error: {e}", flush=True)
            break

        chunk_text = trim_echo(chunk_text)
        dedup = trim_token_overlap(stitched, chunk_text, 64)
        if dedup == chunk_text:
            dedup = trim_char_overlap(stitched, chunk_text, 200)
        aligned = sentence_clip(dedup)
        stitched += aligned

        chunks_done += 1
        print(f">> [GEN] chunk {chunks_done} (+{this_new} tok) — total chars={len(stitched)}", flush=True)

        if srdi_enable:
            if f0 is None and len(stitched) >= CFG.srdi_warmup_chars:
                f0 = build_fingerprint(stitched)
            elif f0 and (chunks_done % CFG.srdi_check_every_chunks == 0):
                s = drift_score(f0, stitched[-min(len(stitched), 1600):])
                if s >= CFG.srdi_drift_threshold:
                    anchor = srdi_anchor(f0, human_style=human_style)
                    anchor_ids = tokenize(anchor)
                    if len(anchor_ids) > CFG.srdi_anchor_max_tokens:
                        anchor_ids = anchor_ids[:CFG.srdi_anchor_max_tokens]
                        anchor = MODEL.detokenize(anchor_ids).decode("utf-8", errors="ignore")
                    carry_prompt = (carry_prompt[-9000:] + "\n" + anchor + "\n")
                    temp = max(0.2, temp + CFG.srdi_temp_nudge)
                    top_p = max(0.5, top_p + CFG.srdi_top_p_nudge)

        carry_prompt = carry_prompt[-10000:] + aligned
        remaining -= this_new

        if not aligned.strip():
            break

    return stitched.strip()

# =========================
# Asset + style builders
# =========================
def llm_dynamic_asset(asset_type: str, topic: str, role: str, qstate_str: str, entropy: float, creativity: float = 0.9, n: int = 6) -> List[str]:
    if GEN_BACKEND == "llama":
        ensure_model_loaded()
    prompt = f"""
For a human-written blog about "{topic}", generate {n} highly original {asset_type}s suitable for the "{role}" role.
Use the quantum signal "{qstate_str[:18]}" as your entropy reference; creativity setting: {(creativity+entropy):.2f}.
Each should be unique, natural, and plausible but not generic. Avoid clichés and stock phrases. Keep items concise.

List (numbered):
1.
""".strip()
    response = generate_chunked(prompt, max_total_new=180, chunk_new_init=90, srdi_enable=True, human_style=True)
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    items = []
    for ln in lines:
        ln = re.sub(r"^\s*(\d+[\.\)]\s*)", "", ln)
        if 2 <= len(ln) <= 140:
            items.append(ln)
    if len(items) < n:
        return llm_dynamic_asset(asset_type, topic, role, qstate_str[::-1], min(1.0, entropy+0.1), creativity, n)
    out = []
    seen = set()
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
        if len(out) >= n: break
    return out

def build_style_profile_llm(topic: str, qstate_vec, entropy: float, creativity: float) -> HumanStyleProfile:
    qstr = format_quantum_state(qstate_vec)
    dms   = llm_dynamic_asset("discourse marker", topic, "Editor", qstr, entropy, creativity, n=7)
    hooks = llm_dynamic_asset("blog section hook", topic, "Writer", qstr, entropy, creativity, n=7)
    openers = llm_dynamic_asset("section opener", topic, "Researcher", qstr, entropy, creativity, n=7)
    personality = {}
    for agent_role in ["Researcher", "Writer", "Editor", "Critic"]:
        trait = llm_dynamic_asset("persona trait", topic, agent_role, qstr, entropy, creativity, n=1)[0]
        personality[agent_role] = trait
    micro_drift = 0.06 + entropy*0.12
    rqf = 0.03 + entropy*0.07
    scp = 0.01 + entropy*0.05
    para_range = (max(2, 3 + int(entropy*2)), max(6, 8 + int(entropy*4)))
    amps = np.abs(qstate_vec[:3])
    amps = amps / (np.sum(amps) + 1e-8)
    mel = (float(0.3 + 0.5*amps[0]), float(0.3 + 0.5*amps[1]), float(0.2 + 0.4*amps[2]))
    return HumanStyleProfile(
        personality=personality,
        discourse_markers=dms,
        hooks=hooks,
        section_openers=openers,
        micro_drift=micro_drift,
        rhetorical_q_freq=rqf,
        self_correction_prob=scp,
        para_len_range=para_range,
        sentence_melody_ratio=mel,
    )

def _qpick(seq: List[str], qstate_vec, salt: int = 0) -> str:
    if not seq: return ""
    base = float(np.sum(np.abs(qstate_vec) * (1+np.abs(np.angle(qstate_vec))))) + salt*0.137
    idx = int(abs(math.sin(base))*1e6) % len(seq)
    return seq[idx]

def human_rhythm_guidance(style: HumanStyleProfile, role: str, qstate_vec) -> str:
    trait = style.personality.get(role, "balanced")
    short, med, long = style.sentence_melody_ratio
    pmin, pmax = style.para_len_range
    markers = ", ".join(style.discourse_markers[:min(5, len(style.discourse_markers))])
    hook = _qpick(style.hooks, qstate_vec, salt=11)
    opener = _qpick(style.section_openers, qstate_vec, salt=23)
    return (
        f"- Personality: {trait}. Sentence melody (short/med/long) ~ {short:.2f}/{med:.2f}/{long:.2f}. "
        f"Paragraph length range: {pmin}-{pmax} sentences.\n"
        f"- Use discourse markers when natural: {markers}.\n"
        f"- Try a hook like: \"{hook}\" or an opener like: \"{opener}\".\n"
        f"- Allow mild micro-drift across sections. Occasional rhetorical question OK."
    )

# =========================
# Agents, prompts, consensus
# =========================
class Agent:
    def __init__(self, name: str, role_preamble: str):
        self.name = name
        self.role = role_preamble
        self.memory: List[str] = []
    def add(self, msg: str):
        msg = msg.strip()
        if msg:
            self.memory.append(msg)
            while len(self.memory) > CFG.mem_depth:
                self.memory.pop(0)
            bank_add(msg)
    def last(self, n: int) -> List[str]:
        return self.memory[-n:] if n <= len(self.memory) else self.memory

def roles():
    return {
        "Researcher": "ROLE: Researcher — gather concrete facts, examples, analogies, and context about {topic}. Prefer practical insights over fluff.",
        "Writer": "ROLE: Writer — craft flowing prose with varied sentence lengths, conversational clarity, and narrative hooks. Use the target tone and perspective.",
        "Editor": "ROLE: Editor — improve structure, headings, transitions, and readability. Reduce repetition and ensure paragraph rhythm.",
        "Critic": "ROLE: Critic — challenge weak claims, ask pointed questions, and push for specific, grounded details and natural voice.",
    }

def make_agents() -> List[Agent]:
    R = roles()
    return [
        Agent("Agent_Researcher", R["Researcher"]),
        Agent("Agent_Writer",     R["Writer"]),
        Agent("Agent_Editor",     R["Editor"]),
        Agent("Agent_Critic",     R["Critic"]),
    ]

def log_event(obj: Dict[str, Any]):
    if not CFG.log_to_file:
        return
    with open(CFG.log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_agent_prompt(agent: Agent, agents: List[Agent], qsignal: str, qstate_vec: np.ndarray, cpu: float, ram: float, entropy: float, stage: str, settings: BlogSettings, style: HumanStyleProfile, title: Optional[str], outline: List[str], section_index: Optional[int], draft_so_far: str, drift_changes: Optional[Dict[str, Any]] = None) -> str:
    others = []
    for other in agents:
        if other is agent:
            continue
        last = other.last(1)[-1] if other.memory else "[no reply]"
        others.append(f"{other.name}: {last}")
    mem_section = "\n".join([f"{agent.name} memory {i+1}: {m}" for i, m in enumerate(agent.last(CFG.mem_depth))])
    other_section = "\n".join(others)
    q_capsules = build_qlrl_capsules(qsignal, qstate_vec)
    target_section = outline[section_index] if (outline and section_index is not None and 0 <= section_index < len(outline)) else "[n/a]"
    role = agent.name.split("_", 1)[-1]
    rhythm = human_rhythm_guidance(style, role, qstate_vec)
    if drift_changes:
        extras = []
        for k, v in drift_changes.items():
            if isinstance(v, str) and v.strip():
                extras.append(f"- Drift {k}: {v.strip()}")
        if extras:
            rhythm = rhythm + "\n" + "\n".join(extras)
    sysblock = f"""
SYSTEM STATUS:
- CPU: {cpu:.2f}%  RAM: {ram:.2f}%  ENTROPY: {entropy:.6f}
- QUANTUM: {qsignal}
- {{quantum_state}} = {qsignal}

Q-LRL RETRIEVAL CAPSULES (quantum-weighted):
{q_capsules}

BLOG META:
- Topic: {settings.topic}
- Audience: {settings.audience}
- Tone: {settings.tone}
- Perspective: {settings.perspective}
- Reading grade: {settings.reading_grade}
- Target words: ~{settings.target_words}
- Title hint: {settings.title_hint or "[none]"}
- Keywords: {settings.keywords or "[none]"}
- Human-style mode: {settings.human_style_mode}
- Stage: {stage}
- Current Title: {title or "[not set]"}
- Target Section: {target_section}

AGENT: {agent.name}
{agent.role}

STYLE GUIDANCE (+drift):
{rhythm}

CONTEXT:
- You are one of four agents collaborating to produce a natural, reader-first blog post.
- Analyze others' latest replies, build on strong points, and keep the voice consistent but lively.
- Prefer clarity and specificity over generic filler.

PEER INSPECTION (latest):
{other_section or "[no peer outputs yet]"}

YOUR RECENT MEMORY:
{mem_section if mem_section else "[empty]"}

DRAFT SO FAR (truncated):
{textwrap.shorten(draft_so_far, width=1200, placeholder=" ...")}

TASK FOR THIS TURN:
- Contribute a focused segment for stage="{stage}".
- If stage is "title", propose 3 compelling title options (H1-ready).
- If "outline", propose a pragmatic outline with {settings.section_count} sections.
- If "section", produce the next section content with subheadings, hooks, and examples.
- If "polish", refine flow, transitions, and add a crisp conclusion.
- Keep it original and natural; avoid repeating section headers verbatim.

Write your contribution now.

GPT4 Correct Assistant:
""".strip()
    if CFG.show_prompts:
        print("\n----- PROMPT START -----\n", sysblock[:1200], "\n----- PROMPT END -----\n")
    return sysblock

def build_consensus_prompt(stage: str, all_replies: Dict[str, str], qsignal: str, settings: BlogSettings, title: Optional[str], outline: List[str], section_index: Optional[int], draft_so_far: str) -> str:
    concat = "\n\n".join([f"{k}:\n{v}" for k, v in all_replies.items()])
    section_name = outline[section_index] if (outline and section_index is not None and 0 <= section_index < len(outline)) else "[n/a]"
    return f"""
CONSENSUS STAGE: {stage}
QUANTUM: {qsignal}
{{quantum_state}} = {qsignal}

GOAL:
- Integrate strongest points.
- Resolve conflicts and remove redundancy.
- Ensure natural human-like rhythm, with varied sentence lengths and clean transitions.
- If stage is "title": Output ONE final H1 title.
- If "outline": Output ONE final outline with exactly {settings.section_count} sections.
- If "section": Output ONE cohesive section for: "{section_name}" (~{max(180, settings.target_words//max(1,settings.section_count))} words).
- If "polish": Output the final edited article, including a short meta description if requested.

BLOG META:
- Topic: {settings.topic}
- Audience: {settings.audience}
- Tone: {settings.tone}
- Perspective: {settings.perspective}
- Reading grade: {settings.reading_grade}
- Target words: ~{settings.target_words}
- Include meta description: {settings.include_meta_description}
- Conclusion style: {settings.conclusion_style}

TITLE (if any): {title or "[not set]"}

OUTLINE (if any):
{chr(10).join(f"- {s}" for s in outline) if outline else "[not set]"}

DRAFT SO FAR (truncated):
{textwrap.shorten(draft_so_far, width=1200, placeholder=" ...")}

AGENT THREADS:
{concat}

Synthesize the final output for this stage now.

GPT4 Correct Assistant:
""".strip()

def apply_quantum_style_drift(draft_so_far: str, qstate_str: str, entropy: float, creativity: float = 0.9) -> Dict[str, Any]:
    prompt = f"""
The current blog draft (excerpt below) has been written with certain style, tone, and rhythm.

Quantum state: "{qstate_str[:22]}", entropy: {entropy:.3f}, creativity: {creativity:.2f}.

INSTRUCTION:
Invent a small, organic, plausible change to the writing voice for the *next* segment/paragraph.
The change should be non-obvious, not explicitly announced, and should affect e.g. pacing, idiom, emotional tone, sentence variety, rhetorical devices, or narrative stance (briefly).

Return a Python dict only, e.g.:
{{"tone": "...", "sentence_structure": "...", "rhetorical_style": "...", "person": "...", "quirk": "..."}}

Excerpt:
{draft_so_far[-900:]}
Next segment changes:
""".strip()
    response = generate_chunked(prompt, max_total_new=120, chunk_new_init=60, srdi_enable=True, human_style=True)
    try:
        first = response.strip().split('\n', 1)[0]
        changes = ast.literal_eval(first)
        if not isinstance(changes, dict): raise ValueError
    except Exception:
        try:
            changes = ast.literal_eval(response.strip())
            if not isinstance(changes, dict): changes = {}
        except Exception:
            changes = {}
    return changes

def run_adversarial_ghostwriter_chain(text: str, topic: str, qstate_str: str, entropy: float, passes: int = 3) -> str:
    draft = text
    for _ in range(max(0, passes)):
        mood = llm_dynamic_asset("editor mood", topic, "Editor", qstate_str, entropy, n=1)[0]
        persona = llm_dynamic_asset("human editor persona", topic, "Editor", qstate_str, entropy, n=1)[0]
        prompt = f"""
Rewrite the passage as if you are {persona} (mood: {mood}).
- Make only subtle, plausible changes (re-ordering, clarity, light rephrasing, gentle humanization).
- Preserve all meaning; do not shorten drastically; keep structure intact.

Passage:
{draft.strip()}

Rewritten:
""".strip()
        draft = generate_chunked(prompt, max_total_new=900, chunk_new_init=260, srdi_enable=True, human_style=True)
    return draft

def run_peer_collab_chatlog(section_prompt: str, topic: str, qstate_str: str, entropy: float, personas: int = 3, n_rounds: int = 4) -> str:
    ppl = llm_dynamic_asset("forum persona", topic, "Collaborator", qstate_str, entropy, n=max(3, personas))
    chatlog = []
    last_message = section_prompt
    for round_idx in range(max(1, n_rounds)):
        persona = ppl[round_idx % len(ppl)]
        mood = llm_dynamic_asset("editor mood", topic, persona, qstate_str, entropy, n=1)[0]
        prompt = f"""
Pretend you are "{persona}" (mood: {mood}). Continue this collaborative writing forum.
- Respond to the last message, suggest rewordings, analogies, or question the structure.
- You may agree, disagree, riff on, or improve the last post.
- A touch of natural banter is acceptable if it aids clarity.

Chat (latest first):
{chr(10).join(chatlog[-6:]) if chatlog else "(start of thread)"}

{persona}: {last_message}

Next reply:
""".strip()
        reply = generate_chunked(prompt, max_total_new=180, chunk_new_init=80, srdi_enable=True, human_style=True).strip()
        chatlog.append(f"{persona}: {reply}")
        last_message = reply
    merge_prompt = f"""
Given the collaborative chat history below, synthesize a single, polished, natural section for the blog,
blending the best ideas and transitions. Keep it cohesive and on-topic.

Chat history:
{chr(10).join(chatlog)}

Final blog section:
""".strip()
    final_section = generate_chunked(merge_prompt, max_total_new=500, chunk_new_init=180, srdi_enable=True, human_style=True)
    return final_section

def parse_titles(raw: str) -> List[str]:
    cands = []
    for line in raw.splitlines():
        line = line.strip(" #-*>\t")
        if not line: continue
        if 5 < len(line) < 140:
            cands.append(line)
    seen = set(); out=[]
    for t in cands:
        if t not in seen:
            out.append(t); seen.add(t)
    return out[:5]

def parse_outline(raw: str, desired_n: int) -> List[str]:
    lines = [re.sub(r"^[\-\*\d\.\)]+\s*", "", l.strip()) for l in raw.splitlines() if l.strip()]
    items = [l for l in lines if 4 < len(l) < 120]
    seen=set(); outline=[]
    for s in items:
        if s not in seen:
            outline.append(s); seen.add(s)
    if len(outline) > desired_n:
        outline = outline[:desired_n]
    return outline

def hsv_to_rgb(h, s, v):
    h = float(h % 360) / 60.0
    i = int(h); f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return int(r*255), int(g*255), int(b*255)

def entropic_loop_metrics(iteration: int = 0) -> Dict[str, Any]:
    entropy, cpu, ram = get_entropy_seed()
    qvec = quantum_state(entropy, cpu, ram, iteration)
    qstr = format_quantum_state(qvec)
    mag = float(np.mean(np.abs(qvec[:32])))
    ang = float(np.mean(np.abs(np.angle(qvec[:32]))))
    hue = (cpu*1.7 + ram*0.9 + mag*360 + ang*180 + iteration*17.3) % 360
    sat = min(1.0, 0.4 + (entropy + mag) * 0.4)
    val = min(1.0, 0.6 + (ang % 1.0) * 0.35)
    rgb = hsv_to_rgb(hue, sat, val)
    base = f"{os.getpid()}|{time.time()}|{cpu:.3f}|{ram:.3f}|{mag:.6f}|{ang:.6f}|{hue:.3f}|{sat:.3f}|{val:.3f}"
    loop_number = int.from_bytes(hashlib.blake2s(base.encode(), digest_size=8).digest(), "big")
    qvar = float(np.var(np.abs(qvec[:64])))
    entropic_gain = (entropy*0.45 + (cpu/100.0)*0.25 + (ram/100.0)*0.2 + min(0.1, qvar))
    generators_between = max(1, min(6, int(1 + math.floor(entropic_gain*10.0))))
    return {"entropy": entropy, "cpu": cpu, "ram": ram, "qvec": qvec, "qstr": qstr, "rgb": rgb, "loop_number": loop_number, "generators_between": generators_between, "entropic_gain": entropic_gain}

def build_origin_dynamic_context(umbrella_topic: str, origin_sentences: List[str], audience_hint: str, tone_hint: str, perspective_hint: str, keywords_hint: str, creativity: float) -> Dict[str, Any]:
    entropy, cpu, ram = get_entropy_seed()
    qvec = quantum_state(entropy, cpu, ram, 777)
    qstr = format_quantum_state(qvec)
    base_prompt = f"""
We are preparing to write several human-feeling blog articles.
Umbrella topic: "{umbrella_topic}"
Audience hint: "{audience_hint}"
Tone hint: "{tone_hint}"
Perspective hint: "{perspective_hint}"
Keywords hint: "{keywords_hint}"
Seed lines:
{chr(10).join("- "+s for s in origin_sentences if s.strip())}
Quantum: "{qstr[:32]}", entropy={entropy:.3f}, creativity={creativity:.2f}

Tasks:
1) Propose a dynamic prompt scaffold template that future generators will use.
2) Propose 5 nested micro-prompts that expand the scaffold into different angles.
3) Propose 5 constraint patterns that subtly shape cadence and idiom without clichés.

Return JSON with keys: scaffold, micro_prompts, constraints.
"""
    resp = generate_chunked(base_prompt, max_total_new=420, chunk_new_init=180, srdi_enable=True, human_style=True)
    scaffold, micros, constraints = "", [], []
    try:
        js = json.loads(resp.strip().splitlines()[0])
        scaffold = js.get("scaffold","")
        micros = js.get("micro_prompts",[])
        constraints = js.get("constraints",[])
    except Exception:
        try:
            js = json.loads(resp)
            scaffold = js.get("scaffold","")
            micros = js.get("micro_prompts",[])
            constraints = js.get("constraints",[])
        except Exception:
            pass
    if not scaffold:
        scaffold = f"Write for {audience_hint} on {umbrella_topic} with a {tone_hint} tone from {perspective_hint}. Infuse topic-specific angles and avoid generic phrasing. Keywords: {keywords_hint}."
    if not micros:
        micros = llm_dynamic_asset("prompt micro-variation", umbrella_topic, "Meta-Composer", qstr, entropy, creativity, n=5)
    if not constraints:
        constraints = llm_dynamic_asset("cadence constraint", umbrella_topic, "Meta-Composer", qstr, entropy, creativity, n=5)
    bank_add(scaffold)
    for m in micros: bank_add(m)
    for c in constraints: bank_add(c)
    return {"scaffold": scaffold, "micro_prompts": micros, "constraints": constraints, "qvec": qvec, "qstr": qstr, "entropy": entropy}

def build_interstitial_shadow_context(topic: str, dynamic_ctx: Dict[str,Any], iteration: int, count: int, creativity: float) -> str:
    met = entropic_loop_metrics(iteration)
    qstr = met["qstr"]
    entropy = met["entropy"]
    rgb = met["rgb"]
    rgb_tag = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    pieces = []
    for i in range(count):
        mp = dynamic_ctx["micro_prompts"][i % len(dynamic_ctx["micro_prompts"])]
        cons = dynamic_ctx["constraints"][i % len(dynamic_ctx["constraints"])]
        prompt = f"""
Scaffold: {dynamic_ctx['scaffold']}
Micro: {mp}
Constraint: {cons}
Color-energy: {rgb_tag}
Quantum: {qstr[:24]} entropy={entropy:.3f}

Write a compact planning note (not final prose) that nudges phrasing, analogies, and transitions for a section about "{topic}". No clichés, no boilerplate. 60-120 words.
Plan:
""".strip()
        note = generate_chunked(prompt, max_total_new=150, chunk_new_init=90, srdi_enable=True, human_style=True)
        pieces.append(note.strip())
    shadow = "\n\n".join(pieces)
    bank_add(shadow)
    return shadow

def run_stage_with_agents(stage: str, agents: List[Agent], settings: BlogSettings, style: HumanStyleProfile, title: Optional[str], outline: List[str], section_index: Optional[int], draft_so_far: str, it_seed: int, drift_changes: Optional[Dict[str, Any]] = None) -> str:
    entropy, cpu, ram = get_entropy_seed()
    qstate_vec = quantum_state(entropy, cpu, ram, it_seed)
    qsignal = format_quantum_state(qstate_vec)
    turn_replies: Dict[str, str] = {}
    for ag in agents:
        prompt = build_agent_prompt(ag, agents, qsignal, qstate_vec, cpu, ram, entropy, stage, settings, style, title, outline, section_index, draft_so_far, drift_changes=drift_changes)
        reply = generate_chunked(prompt, human_style=settings.human_style_mode)
        ag.add(reply)
        turn_replies[ag.name] = reply
        log_event({"stage": stage, "agent": ag.name, "cpu": cpu, "ram": ram, "entropy": entropy, "quantum_state_compact": qsignal, "prompt_preview": prompt[:1500], "reply": reply})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    consensus_prompt = build_consensus_prompt(stage, turn_replies, qsignal, settings, title, outline, section_index, draft_so_far)
    consensus = generate_chunked(consensus_prompt, human_style=settings.human_style_mode)
    bank_add(consensus)
    log_event({"stage": stage, "consensus": consensus, "quantum_state_compact": qsignal})
    return consensus

def llm_generate_new_ideas(base_topic: str, seed_ideas: List[str], qstate_vec, entropy: float, n: int = 5) -> List[str]:
    qstr = format_quantum_state(qstate_vec)
    seeds_txt = "\n".join(f"- {s}" for s in seed_ideas) if seed_ideas else "(none)"
    prompt = f"""
We need {n} fresh, specific blog ideas (titles or one-line summaries) under the umbrella topic "{base_topic}".

Consider these prior ideas:
{seeds_txt}

Quantum signal: "{qstr[:24]}", entropy: {entropy:.3f}.
Make each idea distinct, concrete, and attractive for readers.

List (numbered):
1.
""".strip()
    resp = generate_chunked(prompt, max_total_new=220, chunk_new_init=120, srdi_enable=True, human_style=True)
    out = []
    for ln in resp.splitlines():
        ln = ln.strip()
        ln = re.sub(r"^\s*(\d+[\.\)]\s*)", "", ln)
        if 6 < len(ln) < 160:
            out.append(ln)
    uniq = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq[:n]

def run_blog(settings: BlogSettings, dynamic_ctx: Optional[Dict[str,Any]] = None) -> str:
    entropy, cpu, ram = get_entropy_seed()
    qvec = quantum_state(entropy, cpu, ram, 0)
    style = build_style_profile_llm(settings.topic, qvec, entropy, settings.creativity_bias)
    agents = make_agents()
    title_consensus = run_stage_with_agents("title", agents, settings, style, None, [], None, "", 0)
    titles = parse_titles(title_consensus)
    title = settings.title_hint or (titles[0] if titles else f"{settings.topic}: A Practical Guide")
    outline_consensus = run_stage_with_agents("outline", agents, settings, style, title, [], None, "", 1)
    outline = parse_outline(outline_consensus, settings.section_count)
    if len(outline) < settings.section_count:
        while len(outline) < settings.section_count:
            outline.append(f"Section {len(outline)+1}: Key Idea")
    use_forum_indices = set()
    if settings.use_forum_on_sections.strip().lower() == "all":
        use_forum_indices = set(range(len(outline)))
    elif settings.use_forum_on_sections.strip():
        try:
            use_forum_indices = set(int(x.strip()) for x in settings.use_forum_on_sections.split(",") if x.strip().isdigit())
        except Exception:
            use_forum_indices = set()
    sections_md: List[str] = []
    draft_so_far = f"# {title}\n\n"
    for idx, sec in enumerate(outline):
        meta = entropic_loop_metrics(2+idx)
        k_between = meta["generators_between"]
        shadow = ""
        if dynamic_ctx:
            shadow = build_interstitial_shadow_context(settings.topic, dynamic_ctx, 1000+idx, k_between, settings.creativity_bias)
        qstr_now = format_quantum_state(quantum_state(entropy, cpu, ram, 2+idx))
        drift_changes = apply_quantum_style_drift((draft_so_far + "\n\n" + shadow)[-3000:], qstr_now, entropy, settings.creativity_bias)
        if settings.forum_personas > 0 and (idx in use_forum_indices):
            section_prompt = f"Section target: {sec}\nAudience: {settings.audience}\nTone: {settings.tone}\nPerspective: {settings.perspective}\nConstraints: examples, clarity, gentle humor if appropriate.\nShadow hints:\n{shadow}"
            section_text = run_peer_collab_chatlog(section_prompt=section_prompt, topic=settings.topic, qstate_str=qstr_now, entropy=entropy, personas=settings.forum_personas, n_rounds=max(1, settings.forum_rounds))
        else:
            section_consensus = run_stage_with_agents("section", agents, settings, style, title, outline, idx, draft_so_far + "\n\n" + shadow, 2+idx, drift_changes=drift_changes)
            section_text = section_consensus
        sec_title = re.sub(r"^#+\s*", "", sec).strip()
        section_text = section_text.strip()
        if not section_text.lower().startswith(("##", "# ")):
            section_text = f"## {sec_title}\n\n{section_text}"
        sections_md.append(section_text)
        draft_so_far += section_text + "\n\n"
    final_consensus = run_stage_with_agents("polish", agents, settings, style, title, outline, None, draft_so_far, 2+len(outline))
    meta_description = ""
    if settings.include_meta_description:
        words = re.findall(r"\w[\w\-']*", final_consensus)
        meta_description = " ".join(words[:28]) + ("…" if len(words) > 28 else "")
    front_matter = ""
    if settings.include_front_matter:
        fm = {
            "title": title,
            "description": meta_description.strip(),
            "author": settings.author_name or "Staff Writer",
            "keywords": settings.keywords or "",
            "readingGrade": settings.reading_grade,
        }
        fm_lines = ["---"] + [f"{k}: {v}" for k, v in fm.items()] + ["---", ""]
        front_matter = "\n".join(fm_lines)
    final_md = []
    if front_matter:
        final_md.append(front_matter)
    final_md.append(f"# {title}\n")
    if settings.include_meta_description and meta_description:
        final_md.append(f"> {meta_description}\n")
    final_md.extend(sections_md)
    if not any(re.search(r"(?i)conclusion|wrap\-up|final thoughts", s) for s in outline):
        final_md.append("## Conclusion\n\n" + textwrap.dedent("""\
            The short version? Pick one practical step and try it. Adjust as you go.
            The rest will make more sense when it’s in motion, not on paper."""))
    final_article = "\n\n".join(final_md).strip()
    if settings.ghostwriter_passes and settings.ghostwriter_passes > 0:
        qstr_end = format_quantum_state(quantum_state(entropy, cpu, ram, 7))
        final_article = run_adversarial_ghostwriter_chain(final_article, settings.topic, qstr_end, entropy, passes=settings.ghostwriter_passes)
    return final_article

# =========================
# Entrypoints
# =========================
def main_colab():
    umbrella = COLAB_ORIGIN["umbrella_topic"].strip()
    seeds = [s for s in COLAB_ORIGIN["origin_sentences"] if s.strip()]
    audience = COLAB_ORIGIN["audience_hint"].strip() or "general readers"
    tone = COLAB_ORIGIN["tone_hint"].strip() or "warm, practical, lightly witty"
    perspective = COLAB_ORIGIN["perspective_hint"].strip() or "second-person 'you' with occasional 'we'"
    keywords = COLAB_ORIGIN["keywords_hint"].strip()
    include_fm = bool(COLAB_ORIGIN["include_front_matter"])
    include_meta = bool(COLAB_ORIGIN["include_meta_description"])
    creativity = float(COLAB_ORIGIN["creativity"])
    outfile = COLAB_ORIGIN["outfile"].strip() or "blog_output.md"

    entropy, cpu, ram = get_entropy_seed()
    qvec = quantum_state(entropy, cpu, ram, 99)
    dynamic_ctx = build_origin_dynamic_context(umbrella, seeds, audience, tone, perspective, keywords, creativity)

    topics_for_run = seeds[:max(0, int(COLAB_ORIGIN["count"]))]
    needed = max(0, int(COLAB_ORIGIN["count"]) - len(topics_for_run))
    if needed > 0:
        new_ideas = llm_generate_new_ideas(umbrella, seeds, qvec, entropy, n=needed)
        topics_for_run.extend(new_ideas)
    if not topics_for_run:
        topics_for_run = [f"{umbrella} — Angle {i+1}" for i in range(int(COLAB_ORIGIN["count"]))]

    outputs = []
    for i, t in enumerate(topics_for_run[:int(COLAB_ORIGIN["count"])], start=1):
        settings = BlogSettings(
            topic=t,
            audience=audience,
            tone=tone,
            perspective=perspective,
            reading_grade="8-10",
            target_words=max(600, int(COLAB_ORIGIN["words"])),
            section_count=max(3, int(COLAB_ORIGIN["sections"])),
            human_style_mode=True,
            title_hint=None,
            keywords=keywords,
            author_name=None,
            include_front_matter=include_fm,
            include_meta_description=include_meta,
            forum_personas=max(0, int(COLAB_ORIGIN["forum_personas"])),
            forum_rounds=max(0, int(COLAB_ORIGIN["forum_rounds"])),
            ghostwriter_passes=max(0, int(COLAB_ORIGIN["ghostwriter_passes"])),
            creativity_bias=max(0.1, min(1.5, creativity)),
            use_forum_on_sections=str(COLAB_ORIGIN["forum_on"]).strip(),
        )
        article = run_blog(settings, dynamic_ctx=dynamic_ctx)
        if int(COLAB_ORIGIN["count"]) == 1:
            out_path = outfile
        else:
            root, ext = (outfile, "") if "." not in outfile else (outfile.rsplit(".", 1)[0], "." + outfile.rsplit(".",1)[1])
            out_path = f"{root}_{i}{ext or '.md'}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(article + "\n")
        outputs.append((t, out_path))

    manifest = {
        "base_topic": umbrella,
        "count": int(COLAB_ORIGIN["count"]),
        "seed_ideas": seeds,
        "generated_topics": topics_for_run[:int(COLAB_ORIGIN["count"])],
        "outputs": [{"topic": t, "path": p} for (t, p) in outputs],
        "settings": {
            "audience": audience,
            "tone": tone,
            "perspective": perspective,
            "grade": "8-10",
            "words": int(COLAB_ORIGIN["words"]),
            "sections": int(COLAB_ORIGIN["sections"]),
            "forum_personas": int(COLAB_ORIGIN["forum_personas"]),
            "forum_rounds": int(COLAB_ORIGIN["forum_rounds"]),
            "forum_on": str(COLAB_ORIGIN["forum_on"]),
            "ghostwriter_passes": int(COLAB_ORIGIN["ghostwriter_passes"]),
            "creativity": float(COLAB_ORIGIN["creativity"]),
        }
    }
    with open("run_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    return outputs

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--ideas", type=str, default="")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--audience", type=str, default="general readers")
    parser.add_argument("--tone", type=str, default="warm, practical, lightly witty")
    parser.add_argument("--perspective", type=str, default="second-person 'you' with occasional 'we'")
    parser.add_argument("--grade", type=str, default="8-10")
    parser.add_argument("--words", type=int, default=1200)
    parser.add_argument("--sections", type=int, default=6)
    parser.add_argument("--title_hint", type=str, default=None)
    parser.add_argument("--keywords", type=str, default=None)
    parser.add_argument("--author", type=str, default=None)
    parser.add_argument("--front_matter", action="store_true")
    parser.add_argument("--no_meta_desc", action="store_true")
    parser.add_argument("--outfile", type=str, default="blog_output.md")
    parser.add_argument("--forum_personas", type=int, default=0)
    parser.add_argument("--forum_rounds", type=int, default=0)
    parser.add_argument("--forum_on", type=str, default="")
    parser.add_argument("--ghostwriter_passes", type=int, default=0)
    parser.add_argument("--creativity", type=float, default=0.9)
    args = parser.parse_args()

    seed_ideas = []
    if args.ideas:
        parts = re.split(r"[;\n]", args.ideas)
        seed_ideas = [p.strip() for p in parts if p.strip()]

    entropy, cpu, ram = get_entropy_seed()
    qvec = quantum_state(entropy, cpu, ram, 99)
    dyn = build_origin_dynamic_context(args.topic, seed_ideas, args.audience, args.tone, args.perspective, args.keywords or "", args.creativity)

    topics_for_run: List[str] = []
    topics_for_run.extend(seed_ideas[:max(0, args.count)])
    needed = max(0, args.count - len(topics_for_run))
    if needed > 0:
        new_ideas = llm_generate_new_ideas(args.topic, seed_ideas, qvec, entropy, n=needed)
        topics_for_run.extend(new_ideas)
    if not topics_for_run:
        topics_for_run = [f"{args.topic} — Angle {i+1}" for i in range(args.count)]

    outputs: List[Tuple[str, str]] = []
    for i, t in enumerate(topics_for_run[:args.count], start=1):
        settings = BlogSettings(
            topic=t,
            audience=args.audience,
            tone=args.tone,
            perspective=args.perspective,
            reading_grade=args.grade,
            target_words=max(600, args.words),
            section_count=max(3, args.sections),
            human_style_mode=True,
            title_hint=args.title_hint,
            keywords=args.keywords,
            author_name=args.author,
            include_front_matter=bool(args.front_matter),
            include_meta_description=not args.no_meta_desc,
            forum_personas=max(0, args.forum_personas),
            forum_rounds=max(0, args.forum_rounds),
            ghostwriter_passes=max(0, args.ghostwriter_passes),
            creativity_bias=max(0.1, min(1.5, args.creativity)),
            use_forum_on_sections=args.forum_on.strip(),
        )
        article = run_blog(settings, dynamic_ctx=dyn)
        if args.count == 1:
            out_path = args.outfile
        else:
            root, ext = (args.outfile, "") if "." not in args.outfile else (args.outfile.rsplit(".", 1)[0], "." + args.outfile.rsplit(".",1)[1])
            out_path = f"{root}_{i}{ext or '.md'}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(article + "\n")
        outputs.append((t, out_path))

    manifest = {
        "base_topic": args.topic,
        "count": args.count,
        "seed_ideas": seed_ideas,
        "generated_topics": topics_for_run[:args.count],
        "outputs": [{"topic": t, "path": p} for (t, p) in outputs],
        "settings": {
            "audience": args.audience,
            "tone": args.tone,
            "perspective": args.perspective,
            "grade": args.grade,
            "words": args.words,
            "sections": args.sections,
            "forum_personas": args.forum_personas,
            "forum_rounds": args.forum_rounds,
            "forum_on": args.forum_on,
            "ghostwriter_passes": args.ghostwriter_passes,
            "creativity": args.creativity,
        }
    }
    with open("run_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if os.environ.get("USE_COLAB_INPUTS", "1") == "1":
        main_colab()
    else:
        main_cli()
