# InternVL3 Architecture

Quick reference for subphase profiling.

## ViT-MLP-LLM Structure

```
Image → [ViT] → [MLP connector] → [LLM] → Text
```

- **ViT:** Vision Transformer (InternViT) — patch embed + encoder layers
- **MLP:** Projects vision features into LLM embedding space
- **LLM:** Qwen2.5/Qwen3 decoder — receives interleaved text + image tokens

## Subphases & Code Locations

| Subphase       | File                | NVTX range                 |
|----------------|---------------------|----------------------------|
| Vision encoder | `internvl.py`       | `InternVL_Vision_Encoder`   |
| MLP connector  | `internvl.py`       | `InternVL_MLP_Connector`    |
| LLM prefill    | `gpu_model_runner.py` | `InternVL_LLM_Prefill`    |
| LLM decode     | `gpu_model_runner.py` | `InternVL_LLM_Decode`     |

**Note:** Vision + MLP run during preprocess; LLM prefill/decode split uses `_is_uniform_decode` in the runner.
