"""LLaVA-1.5-7B output-heavy workload (mirrors mllm_sm_vars Generation spec):
128 text input tokens, 256 output tokens. LLaVA-1.5 has a fixed 336x336 image
size — see the inputheavy header for the caveat. Run from a directory
containing eer.jpg.

Honors BATCH_SIZE env (default 1) for batch-sweep reuse.
"""
import os
import torch
import numpy as np
from time import perf_counter
from vllm import LLM, SamplingParams
from PIL import Image

torch.set_grad_enabled(False)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
INPUT_TOKENS = 128
OUTPUT_TOKENS = 256
IMAGE_SIZE = 336
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

print(f"workload=outputheavy batch={BATCH_SIZE} input_tokens={INPUT_TOKENS} "
      f"output_tokens={OUTPUT_TOKENS} image={IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Loading {MODEL_NAME}...")

llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    max_num_seqs=BATCH_SIZE,
    enforce_eager=True,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
)

tokenizer = llm.get_tokenizer()


def build_exact_token_prompt(target_tokens: int) -> str:
    """Pad/trim a seed prompt to exactly target_tokens text tokens."""
    seed = "Explain key ideas in machine learning with clear examples. "
    text = seed * 100
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < target_tokens:
        text += seed
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


PROMPT_TEXT = build_exact_token_prompt(INPUT_TOKENS)
PROMPT = f"USER: <image>\n{PROMPT_TEXT}\nASSISTANT:"

base = Image.open("eer.jpg").convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
base_arr = np.asarray(base, dtype=np.uint8)


def make_variant(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-2, 3, size=base_arr.shape, dtype=np.int16)
    return Image.fromarray(
        np.clip(base_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    )


TOTAL_VARIANTS = (BATCH_SIZE * 5) + 4
images = [make_variant(seed=i) for i in range(TOTAL_VARIANTS)]


def make_batch(start_idx: int):
    return [
        {"prompt": PROMPT, "multi_modal_data": {"image": images[start_idx + j]}}
        for j in range(BATCH_SIZE)
    ]


sampling_params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS)

print(f"--- WARMUP (batch={BATCH_SIZE}) ---")
idx = 0
for w in range(4):
    batch = make_batch(idx)
    idx += BATCH_SIZE
    llm.generate(batch, sampling_params)
    torch.cuda.synchronize()
    print(f"  warmup {w + 1}/4")

print(f"--- PROFILE (batch={BATCH_SIZE}) ---")
batch = make_batch(idx)
torch.cuda.nvtx.range_push("CLOSED_LOOP_INFERENCE")
t0 = perf_counter()
llm.generate(batch, sampling_params)
torch.cuda.synchronize()
elapsed = perf_counter() - t0
torch.cuda.nvtx.range_pop()

print(f"done in {elapsed:.4f}s")
