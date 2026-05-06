# sglang perf uplift on DeepSeek-V4 (AMD MI355X)

Branch: `clean/rocm-deepseek-v4-reorg` (HEAD `637d58f90`)
Base: `a0b59e38d0c09cb17d1a6c0f7923a7386a4a58d9` (upstream sglang AMD foundation)

Total: 45 commits, ~140 files changed, ~28k lines added.

Document path: `/mnt/vast/john/rocm-dynamo/daily-updates/sglang-perf-uplift-on-deepseek-v4.md`

## TL;DR

**What this is**: a code walkthrough of the 45 production commits on the `clean/rocm-deepseek-v4-reorg` branch that bring up DeepSeek-V4 inference on AMD MI355X (gfx950) and progressively close the per-token latency gap to NVIDIA B200.

**Bottom-line result** (Flash-Base FP8, c=4 ISL=OSL=1024 random range_ratio=0.8 N=40, vllm backend):

| | Before | Now | Delta |
|---|---:|---:|---:|
| TPOT median | model didn't run | **20.62 ms** | from-zero |
| Total throughput | n/a | **353.25 tok/s** | from-zero |
| Mid-campaign baseline | ~32 ms / 234 tok/s | 20.62 ms / 353 tok/s | **-35.6% TPOT, +51% throughput** |
| Gap to B200 reference | ∞ | **2.17×** slower | (B200: 9.49 ms / 597 tok/s) |
| Hardware FLOPS gap | none | none | MI355X 2.5 vs B200 2.25 PFLOPS BF16 — gap is software |

**Three waves of work**:
1. **Foundation + bring-up** (commits 1–9): AMD HIP cuda-graph capture fixes, the unified launcher, the CK Tile FP8 sparse-MLA kernel for Pro and Flash, FlyDSL mxfp4 backend for Flash mxfp4.
2. **Stand-up + the major TPOT ships** (commits 10–23): the **two-shot CK V32** decode path (cuts TPOT roughly in half), AITER fp8 MoE backend default, ROCm 7.2 + aiter HEAD migration (HSA dispatch 14.4 → 3.92 µs/launch), HIP RoPE / SWA / expand_seq_lens kernels, V32 fp8 precision uplift.
3. **Trace-driven micro-fusions** (commits 24–45): autotuned sparse-decode dispatch, mHC Triton port, M2 + decode-body MQA prologue megakernels, F4 fused RMSNorm+quant kernel, **compress_decode_old kv_pool fusion** (-3.32 ms TPOT, biggest decode-side ship), hc_pre kernel, paged-MQA-logits + D=512, 10 trace-targeted micro-fusions, fused invalid-mask, fused lonely-q correction, compress_extend_old per-request loop fusion, and the stage1 BLOCK_H=16 tile re-sweep.

**Highest-leverage commits** (read these 8 first to understand ~85% of the perf delta):

| Commit | Title | Why it matters |
|---|---|---|
| 12 | two-shot CK V32 sparse-MLA decode | Cuts decode TPOT roughly in half. The biggest single ship. |
| 16 | enable AITER block-scale fp8 MoE backend by default | Cuts MoE launches 4× per layer. Replaces an entire op category. |
| 21 | migrate to ROCm 7.2 + aiter HEAD | HSA dispatch latency 3.7×; compounds with everything else. |
| 35 | compress_decode_old fused kv_pool kernel | Largest decode-side fusion (-3.32 ms TPOT). |
| 36 | hc_pre decode-shape Triton kernel | Decode-shape kernel replacing torch chain. |
| 39 | trace-targeted micro-fusions | Bundle of 9–10 small Triton kernels driven by trace evidence. |
| 41 | fused lonely-q correction + compress_extend_old fusion | Cascade-pattern fusion of mask construction + materialization. |
| 42 | stage1 BLOCK_H=16 dispatch | Tile re-sweep — prior winner was stale against the post-stack baseline. |

**Where to navigate code**:
- Sparse-MLA hot path (decode): `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`, `python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py`
- Compressor / kv_pool fusions: `python/sglang/jit_kernel/compress_decode_kv_pool_fused_triton.py`, `hc_pre_decode_triton.py`, `m3_indexer_megakernel_triton.py`
- Trace-targeted micro-fusions: `python/sglang/jit_kernel/{fused_invalid_mask,fused_lonely_q_correction,fused_dual_cat,pt_expand,fused_norm_rope,extend_per_request_megakernel,freqs_idx_gather}_triton.py`
- HIP kernels: `python/sglang/srt/layers/csrc/{rope_hip,swa_indices_hip,expand_seq_lens_hip}/`
- Production launcher: `launch_dsv4.sh` (preset = `stacked-best`)

**How to reproduce the SOTA bench**: see [Reproduction recipe](#reproduction-recipe) below for the full container + bind-mount + launcher setup.

```bash
# Inside the rocm720 container with /sgl-pr mounted from the cloned repo:
bash /sgl-pr/launch_dsv4.sh stacked-best     # serves Flash-Base FP8 on TP=4 port 30010
# Then in a second shell, run sglang.bench_serving against http://127.0.0.1:30010
# (full bench command in the Reproduction recipe).
# Expected: TPOT median ~20.6 ms / throughput ~353 tok/s / TTFT median ~248 ms
```

The full per-commit walkthrough is below; see "Where to start reading" for a 30-minute guided path.

## Reading guide

Each commit below has:
- **Hash + Title** — clickable in the GitHub UI
- **Files touched** — focuses on kernels (`jit_kernel/*`, `csrc/*`), attention paths (`layers/attention/`), the main model file (`models/deepseek_v4.py`), and launchers
- **What changed** — short summary of the functional change
- **Perf** — performance signals from the commit body

## Big picture

The 45 commits land in three roughly chronological waves:

1. **Foundation + Pro V32 + Flash mxfp4 bring-up** (commits 1-9) — AMD HIP cuda-graph fixes, the unified launcher, the CK Tile FP8 sparse-MLA kernel for Pro and Flash, and the FlyDSL mxfp4 backend.
2. **Flash-Base FP8 stand-up + the major TPOT ships** (commits 10-23) — investigation work, the two-shot CK V32 path that cuts decode TPOT in half, AITER MoE backend default, HIP RoPE/SWA/expand_seq_lens, ROCm 7.2 + aiter HEAD migration, V32 fp8 precision/mask kernel work.
3. **Trace-driven micro-fusions and the elementwise win cluster** (commits 24-45) — autotuned dispatches, mHC Triton port, the M2/decode-body megakernels, F4 fused RMSNorm+quant kernel, the kv_pool / hc_pre / paged-MQA fusions, and the final lonely-q + invalid-mask + MEGA-3-prime + stage1 BH=16 fusions.

## End-to-end perf evolution (single MI355X host, c=4 ISL=OSL=1024 random range_ratio=0.8 N=40, vllm backend)

| Stage | TPOT median | Throughput total | Notes |
|---|---:|---:|---|
| Pre-foundation (sglang head) | n/a | n/a | model fails to launch |
| Post-foundation + bring-up | crashes / unstable | unstable | bring-up phase |
| After two-shot CK V32 ship | ~42-45 ms | ~80-90 tok/s | first stable production-quality decode |
| After AITER MoE + ROCm 7.2 + HIP kernels | ~32 ms | ~234 tok/s | mid-campaign baseline |
| + kv_pool fusion + hc_pre + Phase 2 | ~24.7 ms | ~285 tok/s | major decode-side ship |
| + paged-MQA-logits + MHC_POST + freqs_idx | ~24.6 ms | ~317 tok/s | trace-driven micro-fusions |
| + T1-T7 + A1/A3/B3/C1 (afternoon stack) | 21.45 ms | 340.82 tok/s | elementwise + invalid-mask |
| + lonely-q + MEGA-3' Stage 1+2 | 20.90 ms | 348.89 tok/s | sparse-attn output cleanup |
| **+ stage1 BH=16 (current SOTA)** | **20.62 ms** | **353.25 tok/s** | tile re-sweep |

B200 reference (same bench config): 9.49 ms TPOT / 597 tok/s. MI355X is 2.17× slower.

---

## Where to start reading (for a new teammate)

**If you have 30 minutes, read these 8 commits in order — they cover ~85% of the perf delta:**

| # | Hash | Title | Why it matters |
|---|---|---|---|
| 12 | `46e051b2a` | DSv4 Flash-Base FP8: two-shot CK V32 sparse-MLA decode | Cuts decode TPOT roughly in half. The single biggest ship. |
| 16 | `cb0457890` | DSv4 Flash-Base FP8: enable AITER block-scale fp8 MoE backend by default | Cuts MoE launches 4× per layer (86 vs 344). Replaces an entire op category. |
| 21 | `fb8a80457` | DSv4 docs/launchers: migrate to ROCm 7.2 + aiter HEAD container image | HSA dispatch latency 14.4 → 3.92 µs/launch (3.7×). Compounds with everything else. |
| 35 | `23ae9c3fb` | DSv4 compress_decode_old: fused kv_pool + gather + APE-add Triton kernel | Single biggest decode-side fusion: -3.32 ms TPOT. |
| 36 | `721927c16` | DSv4 hc_pre decode-shape Triton kernel | Decode-shape kernel replacing torch chain; -0.80 ms TPOT. |
| 39 | `b1e5b13db` | DSv4 Flash-Base FP8: trace-targeted micro-fusions | Bundle of 9-10 small Triton kernels driven by trace evidence. |
| 41 | `9c940713...` | DSv4 Flash-Base FP8: fused lonely-q correction + compress_extend_old fusion | Cascade-pattern fusion (mask construction + materialization). |
| 42 | `be7e7d699` | DSv4 Flash-Base: stage1 BH=16 dispatch | The "tile re-sweep" lever — prior sweeps had stale BLOCK_H winner. |

## Where the production kernels live (for code navigation)

### Sparse-MLA / attention path (the hot path on decode)

| Component | Location | Owner commits |
|---|---|---|
| **CK Tile V32 sparse-MLA decode kernel** (HIP) | `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp` + `mla_decode_fwd.cu` | 2, 5, 8, 12, 22 |
| **CK V32 wrapper + dispatch** | `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py` | 2, 4, 5, 8, 12, 22 |
| **Flash-MLA reference path + adapter** | `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py` | many; see commit 12 (two-shot) for the production path |
| **Triton sparse-attn decode (split-K)** | `python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py` | 24 (autotune dispatch), 42 (BH=16) |
| **Sparse-attn output-cleanup kernel** (lonely-q) | `python/sglang/jit_kernel/fused_lonely_q_correction_triton.py` | 41 |
| **Sparse-MLA Triton work-score combine** | `python/sglang/jit_kernel/sparse_mla_combine_triton.py` (and related) | 17 |

### Compressor / indexer path

| Component | Location | Owner commits |
|---|---|---|
| **compress_decode_old fused kv_pool kernel** | `python/sglang/jit_kernel/compress_decode_kv_pool_fused_triton.py` | 35, 37 |
| **hc_pre decode-shape Triton kernel** | `python/sglang/jit_kernel/hc_pre_decode_triton.py` | 36 |
| **M2 megakernel (compress_decode + extend)** | `python/sglang/jit_kernel/compress_decode_megakernel_triton.py` | 26 |
| **Indexer fused-logits megakernel (M3)** | `python/sglang/jit_kernel/m3_indexer_megakernel_triton.py` | 29, 31 |
| **C1 fused invalid-mask kernel** | `python/sglang/jit_kernel/fused_invalid_mask_triton.py` | 40 |
| **Compress overlap_transform Triton kernel** | `python/sglang/jit_kernel/overlap_transform_triton.py` | 37 |
| **kvscoreold (extend stage 1)** | `python/sglang/jit_kernel/kvscoreold_triton.py` | 41 |
| **Compressor wire-in / model glue** | `python/sglang/srt/layers/attention/compressed/{indexer,compressor,metadata}.py` and `python/sglang/srt/models/deepseek_v4.py` | most commits |

### MoE / quantization path

| Component | Location | Owner commits |
|---|---|---|
| **F4 fused RMSNorm + per-1×128 fp8 quant** | `python/sglang/srt/layers/quantization/fused_rmsnorm_quant.py` | 28 |
| **AITER MoE backend dispatch** | model wire-in via `SGLANG_FORCE_TRITON_MOE_FP8=0`; AITER kernel is in the aiter package | 16 |
| **Hybrid mxfp4 (FP8 attn + mxfp4 experts)** | `python/sglang/srt/layers/quantization/mxfp4.py` + model dispatch | 3 |
| **FlyDSL mxfp4 backend** | external FlyDSL package + wrapper at `python/sglang/srt/layers/quantization/mxfp4.py` | 7 |
| **Pro mxfp4 packed-runtime** | aiter cktile a16w4 dispatch in `mxfp4.py` | 6 |

### Decode-body megakernel (DBM) — Q/K projection chain

| Component | Location | Owner commits |
|---|---|---|
| **DBM Block A: MQA prologue megakernel** | `python/sglang/jit_kernel/decode_body_mqa_prologue_triton.py` (+ Phase 2 scale-applying GEMM autotune) | 34 |
| **kv_write_with_rope megakernel (default-OFF)** | `python/sglang/jit_kernel/m1_kv_write_with_rope_triton.py` | 30, 32 |

### Trace-driven micro-fusion kernels (T1-T7 / A1-A3 / B3 / freqs_idx_gather)

| Component | Location | Owner commits |
|---|---|---|
| **freqs_idx_gather kernel** | `python/sglang/jit_kernel/freqs_idx_gather_triton.py` | 39 |
| **Fused dual-cat for sparse-decode KV+mask (T4)** | `python/sglang/jit_kernel/fused_dual_cat_triton.py` | 39 |
| **pt_expand Triton kernel (A2)** | `python/sglang/jit_kernel/pt_expand_triton.py` | 39 |
| **Fused rmsnorm + RoPE kernel (A1: rope-to-out)** | `python/sglang/jit_kernel/fused_norm_rope_triton.py` | 40 |
| **compress_extend_old per-request loop fusion** | `python/sglang/jit_kernel/extend_per_request_megakernel_triton.py` + `kvscoreold_triton.py` | 41 |
| **fused_invalid_mask construction (C1)** | `python/sglang/jit_kernel/fused_invalid_mask_triton.py` | 40 |
| **invalid_mask publish/lookup helper** | `python/sglang/jit_kernel/invalid_mask_triton.py` | 39 |
| **Phase-G page-table arithmetic + paged-MQA fp8 fused** | `python/sglang/jit_kernel/fp8_paged_mqa_logits_fused_triton.py` + `fp8_paged_mqa_logits_hip.py` | 38, 39 |

### HIP kernels (host-side or sub-kernel optimizations)

| Component | Location | Owner commits |
|---|---|---|
| **HIP RoPE port** | `python/sglang/srt/layers/csrc/rope_hip/apply_rotary_emb.hip` | 15 |
| **HIP make_swa_indices kernel** | `python/sglang/srt/layers/csrc/swa_indices_hip/make_swa_indices.hip` | 18, 20 (default-on flip is in 18) |
| **HIP fused expand_seq_lens kernel** | `python/sglang/srt/layers/csrc/expand_seq_lens_hip/expand_seq_lens.hip` | 20 |
| **mHC Triton port (replaces TileLang PRE+POST)** | `python/sglang/jit_kernel/mhc_pre_triton.py` + `mhc_post_triton.py` | 25 |

### Launchers

| Launcher | Use case | Last touched in |
|---|---|---|
| `launch_dsv4.sh` (root) | **Flash-Base FP8** unified launcher with named presets (`stacked-best`, `stacked-widebs`, etc.) | 41 |
| `launch_dsv4_pro_base.sh` | Pro-Base TP=8 | 27 |
| `launch_dsv4_pro_mxfp4.sh` | Pro mxfp4 TP=8 EP=8 | 21 |

Reproducer for the aligned bench: see [Reproduction recipe](#reproduction-recipe) below.

---

## Reproduction recipe

This is a self-contained recipe to reproduce the SOTA numbers in the TL;DR table on any MI355X host. The recipe was end-to-end verified on chi2774 (2026-05-05) but contains no chi-specific paths or assumptions; any host meeting the prerequisites below will work.

### Prerequisites

| Resource | What you need |
|---|---|
| Hardware | AMD MI355X (gfx950), 288 GB HBM/GPU; TP=4 for Flash/Flash-Base, TP=8 for Pro |
| ROCm driver | rocm 7.0+ kernel driver on the host (the container ships its own ROCm 7.2 user-space, but the host kernel must be recent enough to expose gfx950 via `/dev/kfd` and `/dev/dri`). Verify with `rocm-smi --showproductname` showing `GFX Version: gfx950`. |
| Docker | Any recent docker (≥20.10). User must be able to add `--device /dev/kfd --device /dev/dri --group-add video --group-add render`. |
| Disk | ~250 GB for the container image + ~1.4 TB if pulling all three checkpoints (Flash + Flash-Base + Pro are each ~600 GB unpacked). Single-case reproducers need only one checkpoint. |
| Container image | `rocm/sgl-dev:rocm720-deepseek-v4-mi35x` (ROCm 7.2 + aiter HEAD, public; commit 21 documents why this image is mandatory — HSA dispatch 14.4 → 3.92 µs/launch vs ROCm 7.0). |
| Code repo | `JohnQinAMD/sglang-amd` branch `clean/rocm-deepseek-v4-reorg` at HEAD `637d58f90` (public). |
| Model checkpoints | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Flash-Base`, `deepseek-ai/DeepSeek-V4-Pro` from Hugging Face. Used unmodified in combination with the `sitecustomize.py` shim from Step 2 (the rocm720 image's `transformers` doesn't register `deepseek_v4` directly, so a 3-line runtime alias in `CONFIG_MAPPING` is required; sglang dispatches the DSv4 implementation from `architectures`, not `model_type`). |
| FlyDSL wheel | `flydsl-0.1.3.1+20260418.68f5725-cp310-cp310-...whl` (Flash mxfp4 only — strict version, see commit 7; pre-installed in the rocm720 image). |

### Step 0: pre-flight check

Run on the host before starting:

```bash
# Confirm gfx950 is visible to the kernel + that you can read /dev/kfd
rocm-smi --showproductname | grep -E "GFX Version|Card Series" | head -8
ls -l /dev/kfd /dev/dri/renderD* 2>&1 | head -4

# Confirm docker can pass GPUs through (no actual container yet, just a 1s probe)
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  rocm/dev-ubuntu-22.04:latest rocm-smi --showproductname 2>&1 | tail -10
```

You should see `GFX Version: gfx950` for every card; if rocm-smi shows `Error: amdgpu_get_auth (1) failed (-1)`, your user is missing the `video` or `render` group — fix that first.

### Step 1: clone the code

Pick any working directory on the host (referred to below as `$WORK`). The repo will be bind-mounted into the container at `/sgl-pr`, which is the path the launcher's `PYTHONPATH` and preset comments reference.

```bash
export WORK=$HOME/dsv4-mi355x      # or wherever you have disk + GPU access
mkdir -p "$WORK" && cd "$WORK"

git clone https://github.com/JohnQinAMD/sglang-amd.git sglang_v4_pr
cd sglang_v4_pr
git checkout clean/rocm-deepseek-v4-reorg

# Pin to the documented HEAD so you reproduce the exact numbers in the TL;DR.
git checkout 637d58f90
```

### Step 2: stage the model checkpoints + transformers alias shim

Download the released DSv4 checkpoints from Hugging Face into a directory that will be bind-mounted as `/hf` inside the container. No `config.json` editing or checkpoint conversion is needed — they are used unmodified.

```bash
export HF_DIR=$WORK/hf
mkdir -p "$HF_DIR"

# Install huggingface-cli on the host if you don't have it (any python venv works):
pip install --user "huggingface_hub[cli]"

# If the DSv4 repos are gated for your account, log in once first:
# huggingface-cli login

# Download the case(s) you intend to bench. Each is ~600 GB; pick one to start.
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-Base \
  --local-dir "$HF_DIR/DeepSeek-V4-Flash-Base" --local-dir-use-symlinks False

# Optional, only if you also want Flash mxfp4 / Pro:
# huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
#   --local-dir "$HF_DIR/DeepSeek-V4-Flash" --local-dir-use-symlinks False
# huggingface-cli download deepseek-ai/DeepSeek-V4-Pro \
#   --local-dir "$HF_DIR/DeepSeek-V4-Pro"  --local-dir-use-symlinks False

# After staging:
#   $HF_DIR/DeepSeek-V4-Flash-Base/{config.json, model-*.safetensors, tokenizer*}
#   $HF_DIR/DeepSeek-V4-Flash/...        (optional)
#   $HF_DIR/DeepSeek-V4-Pro/...          (optional)
```

The rocm720 image's `transformers` (5.5.4) doesn't register `deepseek_v4` as a config alias, so `AutoConfig.from_pretrained` would otherwise raise `KeyError: 'deepseek_v4'` before sglang gets a chance to look at `architectures`. Add a tiny `sitecustomize.py` shim — Python imports it automatically at interpreter startup when it's on `PYTHONPATH`:

```bash
mkdir -p "$WORK/_alias"
cat > "$WORK/_alias/sitecustomize.py" <<'EOF'
# Register deepseek_v4 → deepseek_v3 alias in transformers' CONFIG_MAPPING so
# AutoConfig.from_pretrained() can parse the unmodified HF DSv4 release.
# sglang dispatches DSv4 from cfg.architectures (DeepseekV4ForCausalLM),
# not cfg.model_type, so this alias is purely to satisfy AutoConfig.
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    CONFIG_MAPPING.register("deepseek_v4", CONFIG_MAPPING["deepseek_v3"])
except Exception:
    pass
EOF
```

The shim is bind-mounted into the container at `/_alias` in Step 3, and Step 4 prepends `/_alias` to `PYTHONPATH` so `sitecustomize.py` fires before any `import transformers`.

Verified failure mode if the shim is omitted: `ValueError: The checkpoint you are trying to load has model type 'deepseek_v4' but Transformers does not recognize this architecture.` raised inside `AutoConfig.from_pretrained` before sglang even starts loading weights. With the shim active, `AutoConfig` returns `model_type=deepseek_v4 architectures=['DeepseekV4ForCausalLM']` and sglang loads normally.

### Step 3: pull the image and start the container

```bash
docker pull rocm/sgl-dev:rocm720-deepseek-v4-mi35x   # ~88 GB compressed

docker run -d --name sgl-deepseek-v4-mi35x-rocm720 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host \
  --shm-size 64g \
  -v "$WORK/sglang_v4_pr":/sgl-pr \
  -v "$HF_DIR":/hf \
  -v "$WORK/_alias":/_alias \
  rocm/sgl-dev:rocm720-deepseek-v4-mi35x \
  sleep infinity

# Sanity-check from inside the container that everything is visible:
docker exec sgl-deepseek-v4-mi35x-rocm720 bash -lc '
  echo "--- gfx950 visible? ---"; rocm-smi --showproductname | grep "GFX Version" | head -8
  echo "--- mounts ---"; ls /sgl-pr/launch_dsv4.sh /_alias/sitecustomize.py
  ls /hf/DeepSeek-V4-Flash-Base/config.json
'
```

The three bind mounts are load-bearing: `/sgl-pr` and `/hf` are the in-container roots that `launch_dsv4.sh` references directly (`PYTHONPATH=/sgl-pr/python` at the top of the script, `MODEL=/hf/...` from the env), and `/_alias` is where the `sitecustomize.py` shim from Step 2 lives. Do not rename them.

### Step 4: launch the server

The single canonical entry point on this branch is [launch_dsv4.sh](https://github.com/JohnQinAMD/sglang-amd/blob/clean/rocm-deepseek-v4-reorg/launch_dsv4.sh) at the repo root (in-container: `/sgl-pr/launch_dsv4.sh`). It sets `PYTHONPATH=/sgl-pr/python`, exports the full env-knob set for the chosen preset, and runs `python3 -m sglang.launch_server` with TP=4 (Flash/Flash-Base) or TP=8 (Pro). Two case-specific Pro launchers exist as thin wrappers (`launch_dsv4_pro_base.sh`, `launch_dsv4_pro_mxfp4.sh`).

All three invocations prepend `/_alias` to `PYTHONPATH` so the `sitecustomize.py` shim from Step 2 registers the `deepseek_v4` config alias at python startup.

```bash
# Flash-Base FP8 (TP=4): the SOTA configuration
docker exec -d sgl-deepseek-v4-mi35x-rocm720 bash -lc '
  PYTHONPATH=/_alias:${PYTHONPATH:-} \
  PORT=30010 \
  MODEL=/hf/DeepSeek-V4-Flash-Base \
  bash /sgl-pr/launch_dsv4.sh stacked-best > /tmp/sglang_flash-base.log 2>&1
'

# Flash mxfp4 (TP=4 hybrid FP8 attn + mxfp4 experts via FlyDSL)
# NB: must opt out of CK V32 fp8 (commit 22) — set the env explicitly:
docker exec -d sgl-deepseek-v4-mi35x-rocm720 bash -lc '
  PYTHONPATH=/_alias:${PYTHONPATH:-} \
  PORT=30013 \
  MODEL=/hf/DeepSeek-V4-Flash \
  SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0 \
  bash /sgl-pr/launch_dsv4.sh stacked-best > /tmp/sglang_flash.log 2>&1
'

# Pro mxfp4 (TP=8 EP=8)
docker exec -d sgl-deepseek-v4-mi35x-rocm720 bash -lc '
  PYTHONPATH=/_alias:${PYTHONPATH:-} \
  PORT=30014 \
  MODEL=/hf/DeepSeek-V4-Pro \
  bash /sgl-pr/launch_dsv4_pro_mxfp4.sh > /tmp/sglang_pro.log 2>&1
'
```

Wait for the server to be ready (first run JIT-compiles all in-tree Triton + CK kernels and loads the 600 GB checkpoint sharded across 4 GPUs — typically 4-6 min on a clean container, ~75 s on a warm JIT cache):

```bash
docker exec sgl-deepseek-v4-mi35x-rocm720 bash -lc '
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:30010/health >/dev/null 2>&1; then
      echo "ready after $((i*5))s"; exit 0
    fi
    if ! pgrep -f sglang.launch_server >/dev/null 2>&1; then
      echo "ERROR: server died at $((i*5))s — see /tmp/sglang_flash-base.log:"
      tail -60 /tmp/sglang_flash-base.log; exit 1
    fi
    sleep 5
  done
  echo "TIMEOUT after 10 min"; tail -80 /tmp/sglang_flash-base.log; exit 1
'
```

The `stacked-best` preset bakes in the production env knobs (M3 indexer, decode-body Block A, kv_pool fused V1+V2, hc_pre decode Triton, MEGA-3' Stage 1+2, M1 disabled, MHC PRE=0 / POST=1, etc.); see `launch_dsv4.sh:261-380` in the cloned repo.

### Step 5: run the aligned bench

With the server up on port 30010, run `sglang.bench_serving` against it with the B200-aligned config (this mirrors `b200-bench-repro.md` Configs 1+3 — random dataset, range_ratio=0.8, vllm backend, ISL=OSL=1024, c=4, 40 prompts, 5-step profile):

```bash
docker exec -it sgl-deepseek-v4-mi35x-rocm720 bash -lc '
  python3 -m sglang.bench_serving \
    --backend vllm \
    --host 127.0.0.1 --port 30010 \
    --model /hf/DeepSeek-V4-Flash-Base \
    --tokenizer /hf/DeepSeek-V4-Flash-Base \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --num-prompts 40 --max-concurrency 4 \
    --request-rate inf \
    --warmup-requests 8 \
    --seed 1 \
    --profile \
    --profile-num-steps 5
'
# Expected: TPOT median ~20.6 ms / throughput ~353 tok/s / TTFT median ~248 ms
```

For Flash mxfp4 swap port → 30013 + model → `/hf/DeepSeek-V4-Flash`; for Pro swap port → 30014 + model → `/hf/DeepSeek-V4-Pro`.

After the bench, tear down the server:

```bash
docker exec sgl-deepseek-v4-mi35x-rocm720 bash -lc 'pkill -f sglang.launch_server || true'
```

### Critical env knobs (when running a custom server outside `stacked-best`)

The `stacked-best` preset sets these correctly; they are listed here for anyone composing a custom invocation.

| Variable | Flash mxfp4 | Flash-Base FP8 | Why |
|---|:---:|:---:|---|
| `SGLANG_HIP_SPARSE_MLA_DECODE_FP8` | **0 (must)** | 1 (default) | CK V32 fp8 produces garbage tokens at production TOPK on Flash mxfp4 (commit 22). Required ON for Flash-Base FP8. |
| `_M1_KV_WRITE_ROPE` (preset-internal) | 0 | **0 (must)** | M1 megakernel produces garbage tokens E2E on Flash-Base FP8 (commit 30 / 32). |
| `_MHC_PRE` / `_MHC_POST` | 0 / 1 | 0 / 1 | TileLang MHC PRE regressed +21 ms after the 2026-04-29 aiter mhc rebuild; POST is correctness-critical (commit 23). |

### What you should see (Flash-Base FP8 SOTA, sample output)

End-to-end wall time on a warm JIT cache: ~75 s server start + ~3.5 min bench = ~5 min total. First-run on a fresh container adds 3-5 min for kernel JIT compile.

Reference numbers (recipe verified end-to-end on a single MI355X host on 2026-05-05 against the unmodified HF release + `sitecustomize.py` shim, no checkpoint conversion):

```
============ Serving Benchmark Result ============
Backend:                                 vllm
Max request concurrency:                 4
Successful requests:                     40
Benchmark duration (s):                  199.32
Total input tokens:                      36915
Total generated tokens:                  36420
Output token throughput (tok/s):         182.72
Total token throughput (tok/s):          367.93
Median TTFT (ms):                        247.60
Median TPOT (ms):                        20.59
Median ITL (ms):                         19.87
==================================================
```

A reproduction within ±5% on TPOT and TTFT (e.g., TPOT 19.5-21.5 ms, TTFT 235-260 ms) is in-band. Larger deviations:
- TPOT closer to **32 ms / throughput ~244 tok/s** → you are on the mid-campaign baseline. Confirm `stacked-best` preset is selected (the launcher echoes `Preset: stacked-best` at startup) and that the rocm720 image is in use (`docker inspect sgl-deepseek-v4-mi35x-rocm720 | grep Image`).
- TPOT **>40 ms** → likely the wrong image (ROCm 7.0 instead of 7.2; HSA dispatch is 3.7× slower) or another tenant on the GPUs (`rocm-smi` to check).
- TPOT **<18 ms** → almost certainly a different shape (ISL/OSL/c) than the aligned config; re-check the bench-serving args.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'deepseek_v4'` at server start | `sitecustomize.py` shim not loaded | Confirm `/_alias` is in the `docker run` mount list and that the `docker exec` command in Step 4 actually has `PYTHONPATH=/_alias:${PYTHONPATH:-}` set. Verify with `docker exec ... bash -lc 'python3 -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print(CONFIG_MAPPING.get(\"deepseek_v4\"))"'` — should print a `DeepseekV3Config` class, not `None`. |
| `amdgpu_get_auth (1) failed (-1)` from `rocm-smi` | Host user not in `video`/`render` groups, or SELinux blocking `/dev/kfd` | Add user to both groups (`sudo usermod -aG video,render $USER`) and re-login; on RHEL/CentOS check `getenforce`. |
| `gfx950 not detected` / aiter import fails inside container | Wrong container image (e.g. ROCm 7.0 image) | Use `rocm/sgl-dev:rocm720-deepseek-v4-mi35x`; ROCm 7.0 still works but adds +10 ms TPOT (commit 21). |
| Garbage tokens on Flash mxfp4 (Arabic/Chinese unicode) | CK V32 fp8 enabled on Flash mxfp4 | Export `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0` in the `docker exec` env (the Flash mxfp4 invocation in Step 4 does this); re-check it on the running server with `docker exec ... env | grep SPARSE_MLA`. |
| Garbage tokens on Flash-Base FP8 | M1 kv_write_with_rope megakernel re-enabled | Confirm `_M1_KV_WRITE_ROPE=0` in `stacked-best` preset (commit 30 / 32); do not export `SGLANG_M1_KV_WRITE_WITH_ROPE=1`. |
| Server still not ready after 10 min | Weights still loading; first-time JIT compile of all in-tree Triton + CK kernels can take 4-6 min on a clean container | Wait, or `docker exec sgl-deepseek-v4-mi35x-rocm720 tail -f /tmp/sglang_flash-base.log` until you see `Server is fired up`. |
| TPOT plateaus at ~32 ms | Mid-campaign baseline; late-stage env knobs not on | Check `launch_dsv4.sh`'s `stacked-best` case sets `SGLANG_FUSED_COMPRESS_DECODE_KV_POOL=1`, `_V2=1`, `SGLANG_HC_PRE_DECODE_TRITON=1`, `SGLANG_MEGA3_PRIME_*=1`, `SGLANG_DECODE_BODY_BLOCK_A=1`. |
| OOM during weight load | Other process holding HBM, or wrong TP | `rocm-smi` to check for stragglers; restart Docker to clear; ensure TP=4 for Flash/Flash-Base, TP=8 for Pro (the launcher pins this — don't override). |

### B200 reference (for the 2.17× gap claim)

The B200 numbers in the TL;DR table come from the same `--random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.8 --num-prompts 40 --max-concurrency 4` config on an NVIDIA B200 host running upstream sglang main + DeepSeek-V4-Flash-Base: TPOT median 9.49 ms / throughput 597 tok/s / TTFT median ~55 ms. The MI355X gap is 20.62 / 9.49 = **2.17×** TPOT, 597 / 353 = **1.69×** throughput. Hardware peak BF16 FLOPS are within 10% (MI355X 2.5 vs B200 2.25 PFLOPS), so the residual gap is software (launch overhead + remaining elementwise + attn microkernel quality).

---

## Per-commit detail (oldest first)

### 1. `55353021f` — DSv4: AMD HIP cuda-graph capture fixes + unified launcher script

**Kernel files**:
- `python/sglang/jit_kernel/topk_transform_512_triton.py`

**Wire-in**:
- `python/sglang/srt/layers/attention/compressed/indexer.py`
- `python/sglang/srt/layers/attention/compressed/metadata.py`
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Foundation patches for DeepSeek-V4 inference on AMD MI355X:


### 2. `3a49776e1` — DSv4-Pro: CK Tile FP8 sparse MLA decode for V32 (gfx950)

**Kernel files**:
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd.cu`
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`

**Wire-in**:
- `python/sglang/srt/layers/attention/CK_V32_SPARSE_MLA_INTEGRATION.md`
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`

**What changed**:
  Adds a hand-written CK Tile sparse MLA decode kernel for the V32
  (d_v=512 / DSv4-Pro / V3) shape on AMD MI355X. Beats AMD's asm `.co`
  baseline by 1.7-3.4x and the existing Triton fallback by 2.0-4.6x
  across the typical decode B x topk grid (B=1..8, topk=256..2048).

**Perf signals from commit body**:
  - baseline by 1.7-3.4x and the existing Triton fallback by 2.0-4.6x
  - End-to-end at B=1 H=128 topk=512: 47us / 3 kernel launches per call,
  - vs 108us / ~5 launches for asm `.co` and 140us / 20 launches for the


### 3. `75805fbeb` — DSv4-Flash: enable hybrid mxfp4 (fp8 attention + mxfp4 experts) on AMD

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  DeepSeek-V4-Flash ships a hybrid checkpoint: attention/proj weights are
  fp8 e4m3 + ue8m0 block scales (block 128), MoE experts are mxfp4 packed
  int8 + ue8m0 scales (block 32), no expert biases. None of the existing
  quant paths in sglang_v4_pr handled this combination:

**Perf signals from commit body**:
  - out tok/s at isl=256 osl=16.


### 4. `74ce2abed` — DSv4: cache invalid_mask across layers

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`
- `python/sglang/srt/models/deepseek_v4.py`


### 5. `2ba58f99e` — DSv4: template CK V32 sparse-MLA decode on QK_HEAD_DIM

**Kernel files**:
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd.cu`
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`

**What changed**:
  Extend the gfx950 CK Tile FP8 sparse MLA decode kernel to cover both
  production DSv4 shapes through one templated codepath:


### 6. `16586baa4` — DSv4-Pro mxfp4: packed-runtime path via aiter cktile a16w4

**What changed**:
  Loading Pro mxfp4 on AMD MI355X used to OOM during
  process_weights_after_loading: BF16 upcast of 384 routed experts × 61
  layers × hidden 7168 × 2*ipp 6144 × 2 bytes ≈ 386 GB / rank, exceeds
  288 GB / GPU. Without an in-tree packed-mxfp4 MoE runtime, this was
  unfixable.

**Perf signals from commit body**:
  - process_weights_after_loading: BF16 upcast of 384 routed experts × 61
  - layers × hidden 7168 × 2*ipp 6144 × 2 bytes ≈ 386 GB / rank, exceeds
  - output 35.57 tok/s (4.45 tok/s/GPU), TPOT 219.6 ms, TTFT 5583 ms
  - Below Pro-Base R3+A1 (6.51 tok/s/GPU, TPOT 147.84 ms) — cktile a16w4
  - See PRO_MXFP4_BRINGUP.md for the microbench result table, root-cause


### 7. `71029e4b3` — DSv4-Flash mxfp4: FlyDSL backend

**What changed**:
  The cktile a16w4 mxfp4 path goes through aiter.fused_moe → fused_moe_2stages →
  metadata.stage1 → cktile_moe_stage1 → moe_cktile2stages_gemm1, which adds
  significant Python overhead (~150-250 µs per layer call) on top of an
  untuned CK-tile kernel. At Flash decode (c=1, M=1, 6 active experts on TP=4),
  this dominates wall time: 250 ms TPOT vs 115 ms for Flash-Base FP8.

**Perf signals from commit body**:
  - this dominates wall time: 250 ms TPOT vs 115 ms for Flash-Base FP8.
  - cktile (old):  3.95 tok/s output, TPOT 250.7 ms, TTFT 712 ms
  - FlyDSL (new): 19.40 tok/s output, TPOT  50.94 ms, TTFT 427 ms
  - 4.91× output throughput, 4.92× TPOT, 1.67× TTFT
  - For comparison Flash-Base FP8 on the same hardware: 8.61 tok/s, TPOT 115.4 ms.
  - **FlyDSL mxfp4 is now 2.27× faster than FP8 Flash-Base**, finally realizing
  - 1. pip install flydsl-0.1.3.1+20260418.68f5725-cp310-cp310-...whl  (strict version)


### 8. `6856659eb` — DSv4 CK V32 sparse-MLA: parametrize fp8 decode scale

**Kernel files**:
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd.cu`
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`

**What changed**:
  The CK V32 sparse MLA decode kernel was designed assuming KV is stored
  as `torch.float8_e4m3fnuz` (kernel-agents/.../CK_V32_RESULTS.md P0 step:
  "KV switched to fp8_e4m3fnuz"). The decoder applies a hardcoded 0.5×
  fold to compensate the gfx950 `cvt_pk_f32_fp8` HW intrinsic — which
  reads bytes with e4m3fn semantics (bias=7) — back to fnuz semantics
  (bias=8) so the output matches `torch.float8_e4m3fnuz.float()`.

**Perf signals from commit body**:
  - "KV switched to fp8_e4m3fnuz"). The decoder applies a hardcoded 0.5×
  - (correct value) → kernel multiplies by 0.5 → output is **0.5× the true
  - Microbench (microbench_ck_v32_512.py) confirms across both shapes:
  - FlyDSL + Triton sparse_attn_decode (baseline): 19.40 tok/s, 50.94 ms TPOT
  - FlyDSL + CK V32 (now correct):                 19.19 tok/s, 51.48 ms TPOT
  - the speedup we hoped for, but unlocks the previously-broken Flash CK V32


### 9. `34f309060` — DSv4-Flash mxfp4: elementwise fusion stack + HIP attention routing

**Kernel files**:
- `python/sglang/jit_kernel/fused_norm_rope_triton.py`
- `python/sglang/jit_kernel/fused_store_cache_triton.py`
- `python/sglang/jit_kernel/hash_topk_triton.py`
- `python/sglang/jit_kernel/hisparse_transfer_hip.py`
- `python/sglang/jit_kernel/silu_and_mul_masked_post_quant_triton.py`
- `python/sglang/jit_kernel/topk_transform_512_triton.py`

**Wire-in**:
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`
- `python/sglang/srt/layers/attention/sparse_mla_merge.py`
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Ports MI300's 5-patch FP8 attention stack  plus a new HIP-routing
  layer for the JIT C++ kernels that don't compile under HIP. Together
  these deliver Flash mxfp4 c=1 OSL=1024 TPOT 38.96 ms (vs prior 50.94 ms),
  6.42 tok/s/GPU at TP=4, and ~3× over Flash-Base FP8 — the largest
  single-step Flash mxfp4 perf jump in this branch.

**Perf signals from commit body**:
  - these deliver Flash mxfp4 c=1 OSL=1024 TPOT 38.96 ms (vs prior 50.94 ms),
  - 6.42 tok/s/GPU at TP=4, and ~3× over Flash-Base FP8 — the largest
  - paged cache, 7.5-7.9× over torch ref)
  - jit_kernel/benchmark/bench_deepseek_v4.py — microbench harness for
  - TPOT          50.94 ms* → 38.96 ms  (-23.5%)
  - Throughput    19.43 tok/s → 25.66 tok/s  (+32%)
  - Per-GPU       4.85 → 6.42 tok/s/GPU
  - on the multi-split reduce path. microbench_ck_v32_512.py reports


### 10. `b814430a7` — DSv4 CK V32 fp8: investigation sweep + RUN_ALL_MODELS update

**What changed**:
  Updates RUN_ALL_MODELS.md with current CK V32 fp8 status and consolidates
  the investigation microbench under microbench_ck_v32_512.py. Confirms via
  extended sweep that the CK V32 e2e numerical issue does NOT originate in
  the kernel itself; the issue lies in the integration with the Flash mxfp4
  forward path (60 chained-Q layers). Production gates the path off via
  SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0 on Flash mxfp4.

**Perf signals from commit body**:
  - the investigation microbench under microbench_ck_v32_512.py. Confirms via


### 11. `8ec9a820f` — DSv4 Pro decode: router GEMM, fused rmsnorm+RoPE-Q, MQA dispatch, scratch sizing

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Stack of changes that lift Pro-Base and Pro mxfp4 decode performance
  across c=1/8/16 on MI355X (MI355X + MI355X, TP=8) by 12.3–18.2%
  output throughput, 10.9–15.0% lower TPOT, and 39.5–39.9% lower TTFT
  at c=1. Per-GPU at c=8 vs prior published baselines: Pro-Base
  6.51 → 13.07 tok/s/GPU (2.0×); Pro mxfp4 4.45 → 13.00 tok/s/GPU (2.92×).

**Perf signals from commit body**:
  - output throughput, 10.9–15.0% lower TPOT, and 39.5–39.9% lower TTFT
  - 6.51 → 13.07 tok/s/GPU (2.0×); Pro mxfp4 4.45 → 13.00 tok/s/GPU (2.92×).
  - ~96 active tiles. Microbench: 155 → 39 µs (3.96×). Profile-confirmed
  - in production: 187 → 15 µs/call (12.2×), saving 493 ms per 50-step
  - into one Triton kernel. Microbench: 79.65 → 56.11 µs (1.42×) over
  - - Was hardcoded 16384, which forces ~24 GB of eager scratch (8 GB ×
  - fused-Triton kernel is 3.66× SLOWER than the aiter
  - Pro (elementwise stack measured +104% on Flash c=8; on Pro it's a no-op,


### 12. `46e051b2a` — DSv4 Flash-Base FP8: two-shot CK V32 sparse-MLA decode

**Kernel files**:
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_combine_fwd_kernel.hpp`
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd.cu`
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`
- `python/sglang/srt/layers/attention/sparse_mla_merge.py`

**What changed**:
  Two-phase fix to make ``SGLANG_HIP_CK_V32_TWO_SHOT=1`` (the ``extra_k_cache != None``
  two-shot dispatch path) production-viable. Prior attempt (env-only enable) regressed
  TPOT 45.79 → 85 ms because of two compounding issues — both fixed here.

**Perf signals from commit body**:
  - TPOT 45.79 → 85 ms because of two compounding issues — both fixed here.
  - Replaces the (2× CK splitkv + 2× aiter.mla_reduce_v1 + Triton merge_two_sparse_attn_outputs
  - + Triton _sink_fold_inplace_kernel) pipeline with (2× CK splitkv-to-split + 1× CK
  - _sink_fold_inplace_kernel 172 → 0, replaced by mla_combine_fwd_kernel<512> × 164
  - Per-process microbench (microbench_ck_v32_combine.py): 6/7 PASS at cos_sim ≥
  - Diagnosed that combine kernel phase alone landed TPOT at 85.60 ms — same as the regression.
  - regressor was `float8_copy_kernel_cuda` (250 launches × 526 µs = 131.59 ms),
  - not the Triton merge launches combine kernel phase had eliminated.


### 13. `15bdccda4` — DSv4 host-side: fold aten::item GPU syncs into batched async paths

**Wire-in**:
- `python/sglang/srt/layers/attention/compressed/indexer.py`
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Eliminates 536 of 556 aten::item / aten::_local_scalar_dense calls per
  profile window on DSv4 Flash-Base FP8 by removing two anti-patterns:

**Perf signals from commit body**:
  - TPOT median (ms):      42.68       42.64       -0.04 (noise)
  - TTFT median (ms):     230.57      229.73       -0.84 (noise)
  - Output tok/s   :       86.67       86.79       +0.14%
  - Bench duration :      420.22 s   419.65 s      -0.6%
  - aten::item is hygiene, not a TPOT lever, when GPU is already host-bound.


### 14. `63cba7acd` — DSv4-Pro EP=1: piecewise CUDA graph runner

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`
- `python/sglang/srt/layers/attention/compressed/compressor.py`

**What changed**:
  Introduces the canonical sglang piecewise pattern (register_custom_op +
  register_split_op + caller-allocated output) on three free-function leak
  sites in the DSv4 mxfp4 forward path, plus a launcher knob set and aiter
  upstream patches for the torch_compile_guard mutates_args="unknown" bug
  (ROCm/aiter#2780).

**Perf signals from commit body**:
  - TPOT 220 ms (matches iter15 baseline), no MAF, no NCCL timeouts.


### 15. `7362929aa` — DSv4 Flash-Base FP8: HIP RoPE kernel (opt-in)

**Kernel files**:
- `python/sglang/srt/layers/csrc/rope_hip/apply_rotary_emb.cu`

**What changed**:
  Ports `apply_rotary_emb_triton` to a HIP/CK kernel under
  `SGLANG_HIP_ROPE=1` to bypass Triton's per-launch CPU overhead on AMD ROCm.
  Triton's autotune-cache + JIT-cache + arg-serialization dominates per-call
  cost (~194 us measured); the actual GPU rotation is trivial.

**Perf signals from commit body**:
  - - microbench/microbench_rope_hip.py
  - 7-config production-realistic correctness + perf microbench. Validates
  - TPOT median (ms):      42.68         42.76         +0.08 (noise)
  - TTFT median (ms):     230.57        227.58         -2.99
  - Output tok/s   :       86.67         87.11         +0.51%
  - Bench duration :      420.22 s     418.07 s        -0.51%
  - E2E TPOT neutral on this bench config because the GPU is host-bound at
  - launch overhead is hygiene, not a TPOT lever, when GPU is already


### 16. `ca6d0942a` — DSv4 Flash-Base FP8: enable AITER block-scale fp8 MoE backend by default

**What changed**:
  Flips the stacked-best preset to route MoE through aiter::fmoe_bf16_blockscaleFp8_g1u1
  instead of the Triton fmoe path. AITER's variant uses 4× fewer launches per
  layer (86 vs 344) on gfx950 thanks to a fused per-token activation quant +
  group-blocked GEMM kernel.

**Perf signals from commit body**:
  - instead of the Triton fmoe path. AITER's variant uses 4× fewer launches per
  - E2E (MI355X aligned bench): TPOT -1.29 ms / TTFT -11.4 ms / +3.7% throughput.


### 17. `43faf8098` — DSv4 Flash-Base FP8 sparse-MLA: Triton work-score combine kernel

**Kernel files**:
- `python/sglang/jit_kernel/mla_combine_triton.py`

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`

**What changed**:
  Adds an N-way Triton combine kernel for the sparse-MLA split-K reduce
  epilogue, replacing the prior torch-implemented work-score reduction.
  Wires in via SGLANG_FLASHMLA_TRITON_COMBINE=1; production stacked-best
  preset enables this by default.


### 18. `58369c01c` — DSv4 Flash-Base FP8: HIP fused make_swa_indices kernel (opt-in)

**Kernel files**:
- `python/sglang/srt/layers/csrc/swa_indices_hip/make_swa_indices.cu`

**Wire-in**:
- `python/sglang/srt/layers/attention/compressed/paged_prefill.py`

**What changed**:
  Replaces the broken TileLang fast-path tilelang_make_swa_prefill_indices.
  On the gfx950/TVM stack, TileLang silently miscompiles `if cond: return`
  inside @tilelang.jit to an empty function body — get_kernel_source()
  shows the compiled HIP as `extern "C" __global__ void K() {}`. Output
  is uninitialized memory; downstream attention kernels consume garbage
  indices and either IMA (Phase 4b at compress_extend_old) or livelock

**Perf signals from commit body**:
  - reference 65 ms = 1855x speedup).
  - TPOT  41.39 -> 41.43 ms  (neutral; decode untouched)
  - TTFT  219.17 -> 210.16 ms  (-9.0 ms / -4.1%)
  - Output tok/s  89.91 -> 90.58  (+0.74%)
  - Repro: /sgl-pr/microbench/microbench_swa_indices_hip.py


### 19. `96a116eb4` — DSv4 launchers: refresh Flash mxfp4 CK V32 status note

**What changed**:
  Updates the launch_dsv4.sh comment block describing the current
  SGLANG_HIP_SPARSE_MLA_DECODE_FP8 gate state on Flash mxfp4.


### 20. `d15f02901` — DSv4 Flash-Base FP8: HIP fused expand_seq_lens kernel (opt-in)

**Kernel files**:
- `python/sglang/srt/layers/csrc/expand_seq_lens_hip/expand_seq_lens.cu`

**Wire-in**:
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`

**What changed**:
  Replaces the CPU Python loop in paged_prefill.expand_seq_lens (~5 outer
  iterations + per-seq torch.arange + fill_, plus pinned-memory H2D copies)
  with a single GPU dispatch. Works on already-on-device forward_batch.seq_lens
  and extend_seq_lens directly, skipping the .tolist() + tensor allocation +
  .to(device, non_blocking=True) chain.

**Perf signals from commit body**:
  - TPOT  41.29 -> 41.17 ms  (-0.12; noise)
  - TTFT  208.01 -> 208.17 ms (+0.16; noise)
  - Output tok/s  90.74 -> 90.95  (+0.23%)


### 21. `ff11e452a` — DSv4 docs/launchers: migrate to ROCm 7.2 + aiter HEAD container image

**What changed**:
  Switches the deployment image to rocm/sgl-dev:rocm720-deepseek-v4-mi35x and
  upgrades aiter to its HEAD revision. This drops HSA dispatch latency on
  MI355X from 14.4 µs/launch (ROCm 7.0) to 3.92 µs/launch (ROCm 7.2), a
  3.7× reduction that compounds with all subsequent fusion ships.

**Perf signals from commit body**:
  - 3.7× reduction that compounds with all subsequent fusion ships.


### 22. `dd1a5fd00` — DSv4 CK V32 fp8 sparse-MLA: precision uplift + invalid-row mask

**Kernel files**:
- `python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`

**Wire-in**:
- `python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`

**What changed**:
  Consolidates iterative kernel work investigating bf16/FP8 precision
  behavior of the CK Tile FP8 sparse-MLA decode kernel under the Flash
  mxfp4 configuration. The investigation concluded that Flash mxfp4 + V32
  fp8 must remain gated off (SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0) at
  production TOPK because the model trained on torch's bit pattern and
  the kernel cannot match torch within bf16 ULP across 60 chained-Q


### 23. `4fa3cd2fd` — DSv4-Flash mxfp4: disable redundant MHC pre-norm in stacked-best preset

**What changed**:
  Flash mxfp4 doesn't need the MHC pre-norm kernel because its indexer path
  already pre-normalizes the inputs upstream. Running MHC pre-norm wastes
  a cuda-graph slot and inflates TPOT by ~21 ms.

**Perf signals from commit body**:
  - a cuda-graph slot and inflates TPOT by ~21 ms.
  - Validated on MI355X Flash mxfp4 c=4 num=8: TPOT 55.12 → 33.69 ms.


### 24. `ea078a4e2` — DSv4 Triton sparse decode: per-shape autotuned dispatch

**What changed**:
  Sweep of 180 configs/shape on MI355X identified per-shape optimal
  (BLOCK_H, BLOCK_T, BLOCK_D, SPLIT_K, waves_per_eu, matrix_instr_nonkdim)
  for the auto-dispatch-to-split-K branch in triton_sparse_attn_decode.
  Replaces the single hardcoded fallback config that was tuned for one
  (B=1, Topk=512, D_QK=576) shape with a 3-way per-shape dispatch table.

**Perf signals from commit body**:
  - Measured speedups (cuda-graph capture+replay) across all DSv4 shapes:
  - Universal patterns (documented in microbench/triton_sparse_decode_sweep_results.md):
  - <2% gains there, dominated by JIT compile cost. Net TPOT impact estimate


### 25. `940d4bfa9` — DSv4 mHC: Triton port replaces TileLang PRE+POST (default ON)

**Kernel files**:
- `python/sglang/jit_kernel/mhc_post_triton.py`
- `python/sglang/jit_kernel/mhc_pre_triton.py`

**What changed**:
  Replaces sglang's TileLang mHC PRE (gemm_sqrsum splitk + big_fuse) and POST
  (linear-combine across hc_mult slots) with handwritten Triton kernels.
  Gated by SGLANG_MHC_USE_TRITON (default 1); falls through to TileLang on
  unsupported shapes.

**Perf signals from commit body**:
  - ~10x slower per call on Flash-Base FP8 decode. E2E TPOT regressed
  - 32.10 ms → 52.95 ms (-39% throughput on MI355X / MI355X). The Triton
  - - microbench/triton_port_v2/bench_mhc_{pre,post,pre_breakdown}.py: v2
  - microbench framework gates (correctness + production shape + cuda-graph)
  - v2 microbench (MI355X, weighted by production decode shape histogram):
  - - Phase 13 baseline (TileLang, pre-regression):  TPOT 32.10 ms /  244 tok/s/gpu
  - - Default MI355X today (TileLang, regressed):   TPOT 52.95 ms /  144 tok/s/gpu
  - - This commit (Triton, default):                 TPOT 31.27 ms /  250 tok/s/gpu


### 26. `049682f10` — DSv4 compress_decode + compress_extend_old: fused APE-add megakernel

**Kernel files**:
- `python/sglang/jit_kernel/compress_decode_megakernel_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Adds a Triton megakernel for the compress_decode path that fuses
  {compute logits + softmax + APE-add} into a single launch. Two variants:
  - small-S (decode shape): single-pass softmax in registers
  - large-S (extend shape): streaming online-softmax with APE add fused


### 27. `208ddd4b3` — DSv4 Flash-Base FP8: integrate compressor, indexer, scheduler routing

**Kernel files**:
- `python/sglang/jit_kernel/compress_state_triton.py`
- `python/sglang/jit_kernel/fp8_paged_mqa_logits_fused_triton.py`
- `python/sglang/jit_kernel/fp8_paged_mqa_logits_hip.py`
- `python/sglang/jit_kernel/fused_norm_rope_triton.py`
- `python/sglang/jit_kernel/invalid_mask_triton.py`
- `python/sglang/jit_kernel/kvscoreold_triton.py`
- (+3 more)

**Wire-in**:
- `python/sglang/srt/layers/attention/compressed/compressor.py`
- `python/sglang/srt/layers/attention/compressed/indexer.py`
- `python/sglang/srt/layers/attention/compressed/metadata.py`
- `python/sglang/srt/layers/attention/compressed/paged_prefill.py`
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Stands up the Flash-Base FP8 model integration on AMD MI355X:
  - 8 Triton kernel modules referenced by HEAD imports (rmsnorm, fused_rope,
    fused_quant_gemm, mhc_pre/post, etc.)
  - Compressor + indexer wiring in deepseek_v4.py
  - Scheduler routing for the OLD vs NEW compressor paths
  - Launch script env-knob defaults aligned with the in-tree kernel set


### 28. `e8c27e805` — DSv4: fused RMSNorm + per-1x128 fp8 quantization Triton kernel (opt-in)

**What changed**:
  Adds an in-tree Triton kernel that fuses RMSNorm with per-1×128 fp8
  quantization into a single launch. Outputs the (fp8, scale) tuple
  expected by the FP8 GEMM path, so downstream wq_b can consume it
  directly without a separate quant launch.

**Perf signals from commit body**:
  - Adds an in-tree Triton kernel that fuses RMSNorm with per-1×128 fp8
  - microbench framework added to validate bit-equality vs the unfused


### 29. `0277768ac` — DSv4 indexer: fused logits megakernel (Triton)

**Kernel files**:
- `python/sglang/jit_kernel/m3_indexer_megakernel_triton.py`

**Wire-in**:
- `python/sglang/srt/layers/attention/compressed/indexer.py`
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`

**What changed**:
  Replaces the indexer logits chain (multiple small launches per layer)
  with a single Triton megakernel that fuses the chain into one launch.

**Perf signals from commit body**:
  - Replaces the indexer logits chain (multiple small launches per layer)
  - Microbench: 4.43× cuda-graph-replay speedup at production shapes vs


### 30. `a474c8918` — DSv4 Flash-Base FP8: kv_write_with_rope megakernel (default-off)

**Kernel files**:
- `python/sglang/jit_kernel/m1_kv_write_with_rope_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Adds a Triton megakernel that fuses K-RoPE + KV-write into a single
  launch. Bit-exact at the microbench level vs the unfused chain, but
  empirically produces garbage tokens when stacked into production
  forward passes (root cause not yet bisected; suspected interaction
  with the cuda-graph capture/replay timing of dependent ops).

**Perf signals from commit body**:
  - launch. Bit-exact at the microbench level vs the unfused chain, but


### 31. `11134e635` — DSv4 indexer megakernel: BLOCK_L-conditional num_warps tuning

**Kernel files**:
- `python/sglang/jit_kernel/m3_indexer_megakernel_triton.py`

**What changed**:
  Adds per-BLOCK_L tuning for num_warps and waves_per_eu in the indexer
  fused-logits megakernel. Sweep on MI355X production shapes
  shows 1.27× speedup at BLOCK_L=8192 and 1.13× at BLOCK_L=4096.

**Perf signals from commit body**:
  - shows 1.27× speedup at BLOCK_L=8192 and 1.13× at BLOCK_L=4096.


### 32. `a19520450` — DSv4 Flash-Base FP8: revert kv_write_with_rope default-on; enable AITER paged-MQA-logits

**What changed**:
  Two related launcher changes:

**Perf signals from commit body**:
  - megakernel is bit-exact in microbench but produces garbage tokens at
  - single-kernel implementation. -1.99 ms TPOT on Flash-Base FP8 c=4 aligned


### 33. `18f7eabcb` — DSv4 launchers: default SGLANG_HIP_CK_V32_TWO_SHOT to auto

**What changed**:
  Three shipping launchers had SGLANG_HIP_CK_V32_TWO_SHOT=1 hard-coded:
    launch_dsv4_rocm720.sh:77
    launch_dsv4_rocm720_aiter_sampler.sh:66
    start_phase23_c1.sh:19


### 34. `036a175ff` — DSv4 Flash-Base FP8: decode-body MQA prologue megakernel

**Kernel files**:
- `python/sglang/jit_kernel/decode_body_mqa_prologue_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Adds the first decode-body megakernel: fuses the MQA prologue compute
  chain (4 launches) into a single Triton kernel. Includes a scale-applying
  GEMM with autotune sweep over BLOCK_M / BLOCK_N / num_warps to pick
  the best tile shape per (batch, num_heads) shape.

**Perf signals from commit body**:
  - chain (4 launches) into a single Triton kernel. Includes a scale-applying
  - Microbench: 2.55× cuda-graph-replay at production decode shapes.


### 35. `208b3b01c` — DSv4 compress_decode_old: fused kv_pool + gather + APE-add Triton kernel

**Kernel files**:
- `python/sglang/jit_kernel/compress_decode_kv_pool_fused_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Replaces 5+ small GPU operations (advanced index_put + gather + overlap
  shift + APE add) at compress_decode_old:1582-1612 with a single Triton
  kernel. Owns >92% of the index_elementwise budget at production decode
  shapes.

**Perf signals from commit body**:
  - E2E on MI355X Flash-Base FP8 (c=4 aligned bench): TPOT 28.73 → 25.41 ms
  - (-3.32 ms / -11.6%, biggest single-lever win in the campaign);
  - output throughput +12.5%.


### 36. `b6e62e32c` — DSv4 hc_pre decode-shape Triton kernel (default OFF)

**Kernel files**:
- `python/sglang/jit_kernel/hc_pre_decode_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Replaces _hc_pre_torch_impl (deepseek_v4.py:213) at decode shapes (M=1-8) with
  a 3-launch path: (1) Triton per-row RMSNorm + cast → x_flat + rsqrt; (2)
  torch.matmul x_flat @ hc_fn.t(); (3) torch broadcast mul (linear_out * rsqrt).

**Perf signals from commit body**:
  - for M=8192. v1 attempt to extend it for asymmetric Flash regressed +60 ms TPOT.
  - dispatch / 602 calls in eager profile). Estimated -1.5 to -2.0 ms TPOT.
  - Activation pending v2 microbench correctness + graph-replay validation +
  - E2E live smoke (per the M1/B-pre/hc_pre_v1 microbench-pass-but-E2E-fail lessons).


### 37. `02b81bd41` — DSv4 compress_decode_old: absorb overlap_transform into kv_pool kernel

**Kernel files**:
- `python/sglang/jit_kernel/compress_decode_kv_pool_fused_triton.py`
- `python/sglang/jit_kernel/fused_sampler_triton.py`
- `python/sglang/jit_kernel/hc_pre_decode_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Extends the compress_decode_old fused kv_pool kernel to also absorb the
  overlap_transform_decode operation (2 cat launches/layer eliminated).
  The kernel now handles the full kv_pool maintenance + gather + APE-add +
  overlap_transform pipeline in a single Triton launch.

**Perf signals from commit body**:
  - overlap_transform_decode operation (2 cat launches/layer eliminated).
  - E2E delta (vs Phase 1 baseline): TPOT 24.70 → 24.32 ms (-0.38 ms);
  - total throughput +17.3% vs origin.


### 38. `893e9997c` — DSv4 Flash-Base FP8: enable AITER paged-MQA-logits + D=512 head_dim

**Kernel files**:
- `python/sglang/jit_kernel/compress_decode_megakernel_triton.py`

**What changed**:
  Default-on flip for SGLANG_FP8_PAGED_MQA_LOGITS_AITER (single-kernel
  aiter Triton path) plus 4 small fusions identified from trace analysis:
  - fused MHC pre/post entry hygiene
  - AITER QK rmsnorm group_quant default
  - freqs_idx_gather kernel default-on
  - D=512 head_dim support in compress_decode_full (Flash-Base 2604 mode)

**Perf signals from commit body**:
  - E2E on MI355X Flash-Base FP8: TPOT 32.10 → 31.13 ms (-0.97 ms / +3.10% tput).


### 39. `b1414014f` — DSv4 Flash-Base FP8: trace-targeted micro-fusions

**Kernel files**:
- `python/sglang/jit_kernel/freqs_idx_gather_triton.py`
- `python/sglang/jit_kernel/fused_dual_cat_triton.py`
- `python/sglang/jit_kernel/fused_invalid_mask_triton.py`
- `python/sglang/jit_kernel/pt_expand_triton.py`

**Wire-in**:
- `python/sglang/srt/layers/attention/compressed/indexer.py`
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Trace-driven micro-fusions identified from the b200-aligned-bench profile:
  - Page-table arithmetic fold (sub/floor_divide/mul → single Triton kernel)
  - freqs_idx_gather Triton kernel (replaces 4-launch chain at
    compress_decode_old:1739)
  - AITER paged-mqa-logits scratch hygiene
  - Fused dual-cat for sparse-decode KV+mask

**Perf signals from commit body**:
  - TPOT 24.64 → 23.14 ms (-1.50 ms cumulative); +18.4 tok/s throughput.


### 40. `4bd9fd48b` — DSv4 Flash-Base FP8: fused invalid-mask + fused rope-to-out kernels

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Two related ships in one commit:

**Perf signals from commit body**:
  - Predicted yield -0.13 ms; measured -0.81 ms (6× cascade) because
  - bf16_copy launches if upstream ever produces a non-contig input.


### 41. `605e0ea0a` — DSv4 Flash-Base FP8: fused lonely-q correction + compress_extend_old fusion

**Kernel files**:
- `python/sglang/jit_kernel/extend_per_request_megakernel_triton.py`
- `python/sglang/jit_kernel/fused_lonely_q_correction_triton.py`

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Three related ships in one commit:

**Perf signals from commit body**:
  - launches per channel.
  - launches per channel.
  - Per-layer × 60 layers × bs: 16 launches/req → 4+bs total.
  - fused_lonely_q_correction: 2.94-3.94× per call across 6 production
  - Stage 1: 2.49-2.66× at bs=4 mixed-prefill; bit-exact.
  - Stage 2: 6.81-7.14× at production shapes; KV bit-exact, score at
  - TPOT median: 21.46 → 20.90 ms (-0.56 ms / -2.6%)
  - Total throughput: 340.82 → 348.89 tok/s (+7.98 / +2.3%)


### 42. `d7b932113` — DSv4 Flash-Base: stage1 BH=16 dispatch at Topk>=1024 (default-on)

**What changed**:
  Production decode hot path (`_sparse_attn_decode_stage1`) was using
  BLOCK_H=8 from the 2026-04-29 sweep, which only tested up to Topk=512.
  At production T=2048 per-program work is 4x larger and BH=16 amortizes
  H-axis MFMA setup cost decisively.

**Perf signals from commit body**:
  - TPOT median: 20.94 -> 20.62 ms  (-0.32 ms / -1.5%)
  - TPOT mean:   22.24 -> 21.95 ms  (-0.29 ms / -1.3%)
  - Total tput:  348.80 -> 353.25 tok/s  (+4.45 / +1.3%)
  - TTFT median: 247.41 -> 248.31 ms  (within noise)
  - 40 stage1 calls/iter * ~5 us savings = 200 us gross / 0.15 ms TPOT
  - critical-path-adjusted (75% credit) - matches measured -0.32 ms.
  - microbench/triton_sparse_decode_sweep_results.md
  - microbench/sweep_sparse_decode_stage1_v3b.py


### 43. `5afcce2f1` — DSv4: drop CK V32 fp8 sparse-MLA investigation artifacts

**What changed**:
  The CK V32 fp8 sparse-MLA path on the Flash mxfp4 configuration was
  investigated extensively over 23 iterative commits and concluded as
  permanently blocked at the bf16-ULP precision floor; production gates
  the path off via SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0. The diagnostic
  artifacts created during that investigation are no longer required:

**Perf signals from commit body**:
  - - microbench/microbench_ck_v32_combine.py
  - - microbench/microbench_ck_v32_fp8_saturation.py
  - - microbench/microbench_ck_v32_integration.py
  - - microbench/microbench_ck_v32_padded_pool.py
  - - microbench/microbench_ck_v32_perf.py
  - - microbench/microbench_ck_v32_prod_integration.py
  - - microbench/microbench_ck_v32_prod_replay.py
  - - microbench/microbench_ck_v32_score_dump.py


### 44. `b174deda8` — DSv4: replace internal-nickname comments with functional descriptions

**Wire-in**:
- `python/sglang/srt/models/deepseek_v4.py`

**What changed**:
  Comment-only cleanup. Replaces 14 internal-nickname references
  (Phase 24, A2-#1, A2-#2, MEGA-3', Phase A1, Phase 13, etc.) with
  descriptive functional explanations of the surrounding code.


### 45. `637d58f90` — DSv4: drop outdated docs and superseded launcher scripts

**What changed**:
  Cleanup pass to remove documents and scripts that are no longer
  referenced by the current production setup:

