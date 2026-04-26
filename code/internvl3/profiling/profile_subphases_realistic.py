"""InternVL3 realistic workload: long prompt + 1344×1344 image (multi-tile). Run from dir containing eer.jpg."""
import torch
import numpy as np
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

torch.set_grad_enabled(False)

model_name = "OpenGVLab/InternVL3-8B"
print(f"Loading {model_name}...")

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len=4096,                # bumped to fit 500-token prompt + image features + decode
    gpu_memory_utilization=0.80,
    max_num_seqs=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
)

LONG_BASE = (
    "<image>\nYou are an expert architectural and urban historian collaborating with a "
    "graduate research seminar on twentieth- and twenty-first-century built environments. "
    "Your task is to produce a detailed, structured, and rigorous analysis of the "
    "photograph attached above, written for an audience already familiar with basic "
    "architectural vocabulary but not necessarily with the specific structure shown. "
    "Begin by describing the overall composition of the image: identify the dominant "
    "structures, estimate their scale relative to surrounding elements, and explain how "
    "they are arranged within the frame. Note the photograph's vantage point, focal "
    "length, and how those technical choices shape the viewer's perception of mass, "
    "depth, recession, and proportion. Comment on the role of negative space and on any "
    "compositional devices the photographer appears to use (leading lines, framing, "
    "rule-of-thirds, symmetry, asymmetry). "
    "Then transition to the architectural style. Cite specific stylistic markers visible "
    "in the cladding system, the fenestration pattern, the rooflines, the structural "
    "articulation, and any visible ornamentation. If the structure draws on multiple "
    "traditions, identify which periods, regional schools, or named architects each "
    "element appears to reference, and explain the synthesis or tension between those "
    "references. Distinguish between primary stylistic identity and any later additions, "
    "renovations, or contextual modifications that may have altered the original "
    "design intent. "
    "Discuss probable construction materials based on visual texture, color, joint "
    "patterns, and weathering signatures. Distinguish between primary structural "
    "materials, secondary infill or partition systems, and decorative or functional "
    "claddings. Comment on surface patina, oxidation, biological growth, hairline "
    "cracking, and any signs of recent renovation, addition, or partial demolition. "
    "Where evidence is ambiguous, explicitly mark the inference as speculative. "
    "Address the surrounding context with equal care: adjacent buildings, vegetation, "
    "infrastructure (poles, wires, signage), street furniture, pedestrians, and "
    "vehicles that establish scale or suggest the time period. Speculate, with "
    "appropriate hedging, on the building's likely function, the era of construction, "
    "and the city or region in which it stands. Cite the specific visual clues that "
    "drive each inference and explain the chain of reasoning. "
    "Finally, evaluate the photograph itself as a document: lighting conditions, "
    "estimated time of day, weather, depth of field, and how those choices interact "
    "with the subject's legibility and emotional register. Conclude with two or three "
    "open research questions a historian or planner might pose about this structure "
    "and concrete suggestions for how one might begin to answer them. Provide your "
    "complete analysis in six to eight well-developed paragraphs, in clear academic "
    "prose, without bullet points or numbered lists.\n"
    "Question: "
)

TAILS = [
    "What building is this and where was it taken?",
    "What architectural traditions does this structure synthesize?",
    "Estimate the era of construction and justify your reasoning.",
    "Which materials dominate and how are they weathering?",
    "What does the photograph's framing reveal about the photographer's intent?",
]
PROMPTS = [LONG_BASE + t for t in TAILS]

base = Image.open("eer.jpg").convert("RGB").resize((1344, 1344), Image.BICUBIC)
base_arr = np.asarray(base, dtype=np.uint8)

def make_variant(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-2, 3, size=base_arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(base_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))

images = [make_variant(seed=i) for i in range(len(PROMPTS))]

def request(i: int):
    return {"prompt": PROMPTS[i], "multi_modal_data": {"image": images[i]}}

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

# Quick token count for the first prompt
try:
    tk = llm.get_tokenizer()
    n_text_toks = len(tk.encode(PROMPTS[0]))
    print(f"Prompt text tokens (no image): {n_text_toks}")
except Exception:
    pass

print("--- WARMUP ---")
for i in range(4):
    llm.generate(request(i), sampling_params)
    torch.cuda.synchronize()
    print(f"  {i + 1}/4")

print("--- PROFILE (request 4) ---")
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")
t0 = perf_counter()
llm.generate(request(4), sampling_params)
torch.cuda.synchronize()
print(f"done in {perf_counter() - t0:.4f}s")
torch.cuda.nvtx.range_pop()
