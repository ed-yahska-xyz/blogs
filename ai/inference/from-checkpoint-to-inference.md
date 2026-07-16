# From Checkpoint to Conversation

## How models are stored, how llama.cpp / vLLM / MLX load them, and how your text actually reaches the model

Training gets all the attention, but everything users experience happens *after* training ends. This document follows a model from the moment training finishes to the moment it streams tokens back at you, through the lens of three inference engines: **llama.cpp**, **vLLM**, and **MLX**.

---

## 1. The moment after training

Pretraining, fine-tuning, and RLHF differ enormously in *how* they update a model — but they all produce the **same kind of artifact**. When the last optimizer step runs, what exists is:

- **Weights** — hundreds of named tensors (a "state dict"): embedding matrices, attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), MLP layers, norms, and the output head. For a 7B-parameter model in 16-bit floats, that's ~14 GB of raw numbers.
- **Architecture config** — a small file (`config.json`) declaring the shape those tensors plug into: number of layers, hidden size, attention head counts, vocabulary size, RoPE parameters, context length.
- **Tokenizer** — the vocabulary and merge rules (`tokenizer.json`, `tokenizer_config.json`) that map text ↔ integer IDs. The model never sees text; it only sees these IDs.
- **Chat template** — for instruction/RLHF-tuned models, a Jinja template describing how to wrap a conversation in the special tokens the model was trained on (e.g. `<|im_start|>user ... <|im_end|>`). Get this wrong and a perfectly good model behaves badly.

Crucially, **weights alone are not a runnable thing**. They're inert data. An inference engine is the program that reconstructs the computation graph from the config, fills it with the weights, and executes it.

> Training artifacts vs. distribution artifacts: labs also checkpoint optimizer state (Adam moments, LR schedules) during training, which can triple the size. That's stripped before distribution — you ship only what inference needs.

---

## 2. How models are stored and distributed

### 2.1 The legacy: pickled PyTorch (`.bin`, `.pt`)

The original way to save a model was `torch.save()`, which uses Python's **pickle**. Pickle serializes arbitrary Python objects — including code. Loading a pickled file can therefore **execute arbitrary code**: a malicious checkpoint can read your environment variables or download a payload the moment you load it ([safetensors GitHub](https://github.com/safetensors/safetensors), [DataCamp overview](https://www.datacamp.com/blog/safetensors-format)). Pickle files also can't be partially read — you deserialize everything to get anything. The ecosystem has largely moved on.

### 2.2 The standard: safetensors

[**safetensors**](https://huggingface.co/docs/safetensors/index) is Hugging Face's answer and now the default format on the Hub. The format is almost embarrassingly simple, and that's the point:

```
[ 8 bytes: N = header size (u64, little-endian) ]
[ N bytes: JSON header                          ]
[ raw tensor bytes, back to back                ]
```

The JSON header maps each tensor name to its dtype, shape, and byte offsets:

```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [0, 33554432]
  },
  ...
}
```

Properties that matter for inference:

- **No code execution** — it's data and a JSON index, nothing else.
- **Zero-copy / mmap-friendly** — because offsets are declared up front, an engine can memory-map the file and read tensors lazily, or DMA them straight to GPU. Hugging Face reports loading BLOOM onto 8 GPUs went from ~10 minutes (pickle) to ~45 seconds ([safetensors docs](https://huggingface.co/docs/safetensors/index)).
- **Sharding** — big models are split into `model-00001-of-00004.safetensors` etc., with a `model.safetensors.index.json` mapping each tensor name to its shard.

A model on the Hugging Face Hub is thus a **directory**: sharded safetensors + `config.json` + tokenizer files + chat template. This is the native input for both vLLM and MLX.

### 2.3 The single-file format: GGUF

llama.cpp takes a different philosophy. [**GGUF**](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) ("GGML Universal File") packs *everything* — weights, architecture metadata, tokenizer vocabulary, even the chat template — into **one binary file**:

```
[ magic "GGUF" | version (v3)                    ]
[ tensor count | metadata KV count               ]
[ metadata key-values:                           ]
[   llama.block_count, llama.context_length,     ]
[   tokenizer.ggml.tokens, tokenizer.chat_template, ... ]
[ tensor info: name, shape, type, offset         ]
[ tensor data, 32-byte aligned for mmap          ]
```

Two design goals drive this ([format guide](https://apxml.com/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gguf-format), [deep dive](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)):

- **Self-containment.** `ollama run` or `llama-cli -m model.gguf` needs no Python, no config directory, no Hub download of a separate tokenizer. One file is the whole model. The extensible key-value metadata means new architectures don't break old files.
- **Quantization as a first-class citizen.** GGUF tensors can be stored in block-quantized types — `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`, and newer importance-weighted `IQ` variants — where weights are compressed to ~4–6 bits with per-block scale factors. A 7B model shrinks from 14 GB to ~4 GB, which is what makes laptop and phone inference practical. Different tensors in one file can use different types (e.g. keep the output head at higher precision).

The typical pipeline is: download Hub safetensors → `convert_hf_to_gguf.py` → `llama-quantize` to the target bit-width. Tensor data is aligned to 32-byte boundaries specifically so the file can be **memory-mapped** and used in place — more on that below.

### 2.4 MLX: safetensors, rearranged for Apple Silicon

[MLX](https://github.com/ml-explore/mlx) (Apple's array framework) doesn't invent a new container. An MLX model — e.g. anything under the [mlx-community](https://huggingface.co/mlx-community) Hub organization — is a directory of **safetensors + config.json + tokenizer files**, structurally the same as a Hub model, with two differences:

- Weights are pre-converted (and usually pre-quantized, e.g. 4-bit with group-wise scales) into the layout `mlx-lm`'s model classes expect.
- Quantization parameters live in `config.json` (`"quantization": {"group_size": 64, "bits": 4}`).

Conversion is one command: `mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.3 -q`.

### 2.5 Summary

| | Hub safetensors | GGUF | MLX |
|---|---|---|---|
| Container | directory (shards + JSON sidecars) | **single binary file** | directory (safetensors + JSON) |
| Tokenizer/template | separate files | **embedded in file** | separate files |
| Typical precision | BF16/FP16 (or FP8) | block-quantized 2–8 bit | 4-bit / 8-bit MLX quant |
| Primary consumer | vLLM, transformers, MLX | llama.cpp, Ollama, LM Studio | mlx-lm on Apple Silicon |
| Code execution risk | none | none | none |

---

## 3. How each engine loads a model

Loading is where the three engines' philosophies diverge: **map it** (llama.cpp), **place it** (vLLM), **share it** (MLX).

### 3.1 llama.cpp — mmap and go

llama.cpp is a dependency-free C/C++ engine built on the ggml tensor library, designed to run anywhere.

1. **Parse the GGUF header and metadata.** From the KV pairs it learns the architecture (`llama`, `qwen2`, ...), layer count, and hyperparameters, and reconstructs the tokenizer entirely from the file.
2. **`mmap` the tensor data.** Instead of reading 4 GB into allocated RAM, it memory-maps the file: the OS pages weights in on first touch, startup is near-instant, and the page cache means a second process reuses the same physical memory ([GGUF format notes](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format)). The 32-byte alignment in the format exists precisely for this.
3. **Optionally offload layers to a GPU.** With `-ngl N`, the first N transformer blocks are copied to GPU memory (Metal, CUDA, Vulkan, ROCm); the rest run on CPU. This layer-split is why llama.cpp gracefully handles models bigger than VRAM.
4. **Allocate the KV cache and compute buffers** sized from the metadata (context length × layers × heads).

Quantized weights are *not* dequantized at load. The compute kernels do fused dequantize-and-multiply on the fly, so memory footprint stays at the quantized size.

### 3.2 vLLM — build the engine around the GPU

[vLLM](https://vllm.ai/) is a Python/CUDA serving engine from UC Berkeley's Sky Computing Lab, built for **throughput on datacenter GPUs** ([PagedAttention paper, SOSP 2023](https://en.wikipedia.org/wiki/VLLM)). Loading (`vllm serve meta-llama/Llama-3.1-8B-Instruct`) looks like:

1. **Resolve the model** from the Hub (or a local path) and read `config.json`; the `architectures` field selects one of vLLM's reimplemented model classes (it does not run the model's own Python code).
2. **Stream safetensors shards into GPU memory**, sharding tensors across GPUs if tensor parallelism is enabled (`--tensor-parallel-size`). Safetensors' declared offsets make this a straight copy, no deserialization.
3. **Profile memory, then pre-allocate the rest of VRAM as a paged KV cache.** This is vLLM's signature: instead of reserving contiguous per-request KV memory (which fragments badly), it carves free VRAM into fixed-size **blocks** and hands them to requests on demand — virtual memory for attention ([PagedAttention explained](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching)).
4. **Capture CUDA graphs / warm up kernels** so per-token launch overhead is minimal.

The result is a resident server process: model pinned in VRAM, KV-cache pager running, scheduler ready to interleave hundreds of concurrent requests via **continuous batching** (new requests join the running batch every step; finished ones leave immediately).

### 3.3 MLX — load into memory both processors share

On Apple Silicon the CPU and GPU share one physical **unified memory** pool, and MLX is designed around that fact ([WWDC25: Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)):

1. `mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")` reads `config.json`, instantiates the model class, and loads the safetensors weights.
2. Arrays land in unified memory — there is **no host→device copy step at all**. The same buffers are visible to CPU and GPU; MLX schedules ops on either without transfers.
3. MLX is **lazily evaluated**: loading and even building the forward pass constructs a computation graph; nothing actually computes until evaluation is forced during generation. This keeps load cheap and lets MLX fuse operations.

There's no server machinery by default (though `mlx_lm.server` exists) — MLX's center of gravity is the local, single-user Python/Swift process.

---

## 4. From your text to the first token

Whichever engine you use, your string goes through the same five-stage pipeline. The engines differ in *where* you inject the text.

### 4.1 The pipeline

```
"What is inference?"                                    (your text)
        │  1. chat template (Jinja)
        ▼
"<|im_start|>user\nWhat is inference?<|im_end|>\n<|im_start|>assistant\n"
        │  2. tokenize (BPE)
        ▼
[151644, 872, 198, 3838, 374, 44378, 30, 151645, ...]   (token IDs)
        │  3. PREFILL — one big parallel forward pass over the whole
        ▼      prompt; fills the KV cache; compute-bound
logits over the vocabulary for the *next* token
        │  4. DECODE — sample a token, append it, run one step,
        ▼      repeat; reads the whole KV cache each step; memory-bound
[..., 641, 2202, ...]                                   (generated IDs)
        │  5. detokenize + stream
        ▼
"Inference is..."
```

Two details worth internalizing:

- **The chat template is part of the model contract.** RLHF taught the model to respond inside a specific token scaffold. All three engines apply it for you at the chat-API level (llama.cpp reads it from GGUF metadata; vLLM and mlx-lm from `tokenizer_config.json`), but raw completion APIs skip it — a classic source of "why is this model bad" bugs.
- **Prefill and decode are different workloads.** Prefill processes thousands of tokens in parallel (compute-bound, fast per token); decode generates one token at a time, re-reading the KV cache each step (memory-bandwidth-bound). This split is why the KV cache dominates inference engineering — it's exactly what PagedAttention (vLLM), quantized KV options (llama.cpp), and unified memory (MLX) are each optimizing in their own way.

### 4.2 Sending text to llama.cpp

```bash
# CLI, one-shot
llama-cli -m qwen2.5-7b-instruct-q4_k_m.gguf -p "What is inference?" -ngl 99

# Server: OpenAI-compatible HTTP
llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080
curl http://localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "What is inference?"}],
  "stream": true
}'
```

`llama-server` applies the GGUF-embedded chat template, tokenizes in C++, and streams tokens back as server-sent events. There's also a C API (`llama_tokenize`, `llama_decode`) that bindings in every language wrap.

### 4.3 Sending text to vLLM

```python
# Offline batch API
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
out = llm.chat([{"role": "user", "content": "What is inference?"}],
               SamplingParams(max_tokens=256))
```

```bash
# Serving: OpenAI-compatible, drop-in for the openai SDK
vllm serve meta-llama/Llama-3.1-8B-Instruct
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [{"role": "user", "content": "What is inference?"}]
}'
```

Each incoming request is templated and tokenized, then handed to the scheduler, which allocates KV-cache pages and slots the request into the **continuous batch** — your prompt's prefill runs alongside hundreds of other users' decode steps on the same GPU ([vLLM docs](https://vllm.ai/)).

### 4.4 Sending text to MLX

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is inference?"}],
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256))
```

```bash
# or the CLI / an OpenAI-compatible server
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
                --prompt "What is inference?"
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

Note that mlx-lm makes the template step *explicit* (`apply_chat_template`) in the Python API — a nice pedagogical property, since you can print exactly what the model will see.

### 4.5 Inside the forward pass: matmuls all the way down

Strip away the formats, the servers, and the schedulers, and what an inference engine *computes* is remarkably simple to state:

> **A stack of matrix multiplications interleaved with cheap nonlinearities, run in parallel across tokens, producing a probability distribution over the vocabulary — from which a sampler, shaped by temperature, picks the next token.**

That's the whole game. Every optimization in this document — quantization, mmap, PagedAttention, unified memory — exists to feed these matmuls faster or with less memory. Here is one decode step end to end:

```
 token IDs ──► embedding lookup                              (row-select from a weight matrix)
                     │
                     ▼
 ┌── transformer block × N ── sequential in depth, parallel across tokens & batch ──┐
 │                                                                                  │
 │   norm ──► Q = x·Wq    K = x·Wk    V = x·Wv           weights × activations      │
 │                    │        │          │                                         │
 │                    │        └──────────┴──► appended to KV cache                 │
 │                    ▼                                                             │
 │   attention:  softmax( Q·Kᵀ / √d ) · V           activations × activations ←──   │
 │                    │                              (the matmul the KV cache feeds)│
 │                    ▼                                                             │
 │   project (·Wo) ──► add residual                                                 │
 │   norm ──► MLP:  SiLU(x·Wgate) ⊙ (x·Wup) ──► ·Wdown ──► add residual             │
 │                                                                                  │
 └──────────────────────────────── repeat N times ──────────────────────────────────┘
                     │
                     ▼
 final norm ──► logits = h · Wunembed        one score per vocab entry (~128K numbers)
                     │
                     ▼
 logits ÷ temperature ──► softmax ──► probability distribution over next token
                     │
                     ▼
 sampler (greedy / top-k / top-p) ──► next token ──► append to sequence, loop
```

And what temperature actually does — it rescales the logits *before* softmax, sharpening or flattening the distribution the sampler draws from:

```
 logits for next token:   "infer" 4.1   "reason" 3.9   "think" 2.0   "guess" 1.2

 T → 0    infer ████████████████████▏            near-argmax: always the top token
 T = 0.7  infer ████████████▏ reason ██████▊ think ▊          confident but varied
 T = 1.0  infer █████████▎ reason ███████▋ think █▏ guess ▍   the model's native distribution
 T = 2.0  infer █████▍ reason █████ think ███▏ guess ██▍      flattened: creative / erratic
```

Three refinements that make the simple statement exactly right:

1. **"In parallel" is across tokens and batch, not depth.** During prefill, all prompt tokens flow through each layer simultaneously (one big matmul); in vLLM, whole batches of users do. But the N layers themselves run strictly in sequence — layer 12 needs layer 11's output. Depth is the serial dimension; width is the parallel one.
2. **Not every matmul involves weights.** The attention score computation (`Q·Kᵀ` and `·V`) multiplies *activations by activations* — data by data. That's precisely why the KV cache exists: K and V are recomputable-but-expensive activations worth memoizing, and why attention, not the MLP, is what PagedAttention, sliding windows, and attention sinks all target.
3. **Temperature shapes; the sampler picks.** Temperature is applied to the logits before softmax (÷T), controlling how peaked the distribution is — but the actual selection is a separate step (greedy argmax at T→0, or top-k/top-p/min-p filtered sampling otherwise). Temperature is a dial on the distribution, not the picker itself.

---

## 5. Three engines, one shape, three centers of gravity

| | llama.cpp | vLLM | MLX (mlx-lm) |
|---|---|---|---|
| Format consumed | GGUF | Hub safetensors | MLX-converted safetensors |
| Load strategy | mmap, lazy page-in, optional GPU offload | eager copy to VRAM, paged KV pre-allocation | load into unified memory, lazy graph |
| Killer feature | runs quantized models anywhere, zero deps | PagedAttention + continuous batching for many users | zero-copy CPU/GPU on Apple Silicon |
| Sweet spot | edge, laptops, single-file portability | multi-user GPU serving at scale | local dev on Macs |
| Text input | `llama-cli` / OpenAI-compatible `llama-server` / C API | `LLM.generate()` / OpenAI-compatible `vllm serve` | Python API / `mlx_lm.generate` / `mlx_lm.server` |

The through-line: a trained model is just tensors plus a contract (config, tokenizer, template). Formats differ in how they package that contract — one portable file vs. a directory of standards — and engines differ in which scarce resource they organize around: **disk→RAM paging** (llama.cpp), **VRAM** (vLLM), or **unified memory** (MLX). The text pipeline — template, tokenize, prefill, decode, detokenize — is identical everywhere; once you see it in one engine, you can read all three.

---

## 6. Where each engine belongs: laptop, small service, data center

If you've tried running all three locally, you've probably noticed llama.cpp and MLX feel effortless while vLLM feels heavy. That's not a tuning problem — it's each engine's design assumptions showing through. The question an engine really answers is: **how many users am I built for?**

### 6.1 Why vLLM feels heavy locally (it's supposed to)

vLLM's very first act on startup is to **claim ~90% of your GPU's memory** — `gpu_memory_utilization` defaults to `0.9` — and turn everything left over after the weights into a pre-allocated paged KV-cache pool ([vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/), [tuning guide](https://devopsbeast.com/blog/vllm-gpu-memory-utilization)). That pool is the whole point: its size determines how many *concurrent* sequences the scheduler can keep in flight.

One nuance to the "more RAM for speed per token" intuition: the RAM doesn't buy *per-token latency* — a single stream on vLLM isn't dramatically faster per token than llama.cpp on the same GPU. It buys **aggregate throughput**: tokens per second summed across dozens or hundreds of simultaneous requests sharing one forward pass via continuous batching. If you're the only user, you're paying the memory rent of a hundred-seat restaurant to eat alone. Add that vLLM assumes CUDA-class GPUs (there's no first-class Metal path on a Mac), plus a slow startup phase (memory profiling, CUDA graph capture) and a Python/driver stack, and the local experience you had is exactly the predicted one.

llama.cpp and MLX invert every one of those assumptions: mmap or unified memory instead of pre-allocation, one (or a few) streams instead of hundreds, instant start, no resident server required.

### 6.2 The three deployment tiers

**Local machine (one user, your hardware).**
The scarce resources are RAM and battery, concurrency is 1, and cold-start matters because the model isn't always resident. This is llama.cpp and MLX territory:
- **llama.cpp** if you need portability — any OS, any GPU vendor or none, one GGUF file, memory footprint capped by the quantization you chose. It's the engine inside Ollama and LM Studio for a reason.
- **MLX** if you're on Apple Silicon — unified memory means a 32 GB MacBook genuinely runs 30B-class models, with no VRAM-vs-RAM split to manage at all.
- **vLLM** here is the wrong tool: it will grab most of your GPU for a KV pool serving users who don't exist.

**Small service (a team or app — tens of concurrent users, one box).**
The interesting middle tier: you need an always-on OpenAI-compatible endpoint and *some* concurrency, but not datacenter scale. All three can do it, with different flavors:
- **llama.cpp** — `llama-server` supports parallel slots and continuous batching; a single consumer GPU (or even a strong CPU box) serves an internal tool fine. Great when the budget is one machine and the model is quantized.
- **MLX** — `mlx_lm.server` exposes an OpenAI-compatible API with continuous batching, which makes a Mac mini or Mac Studio a legitimately good small-team inference box; multi-Mac setups over `mx.distributed` push this further ([WWDC26: local agentic AI with MLX](https://developer.apple.com/videos/play/wwdc2026/232/)). Compelling economics: high memory-per-dollar, low idle power, and the privacy of on-prem.
- **vLLM** — this is the tier where it starts to earn its memory. If you have one proper CUDA GPU (A10, L40S, 4090) and real concurrent traffic, vLLM's scheduler will beat both others on requests-per-second — just tune `gpu_memory_utilization` down from 0.9 if the box is shared.

**Data center (thousands of users, fleets of GPUs).**
Now the scarce resource is dollars-per-token at scale, and everything llama.cpp and MLX skip becomes mandatory: paged KV memory to pack requests densely, continuous batching to keep GPUs saturated, tensor/pipeline parallelism to span GPUs and nodes, FP8, prefix caching, speculative decoding, metrics. This is what vLLM (and peers like TensorRT-LLM and SGLang) is *for* — it's the reference open-source serving stack for exactly this tier ([vllm.ai](https://vllm.ai/)). llama.cpp doesn't scale out this way, and Apple hardware isn't racked this way (yet).

| | Local (1 user) | Small service (~10s of users) | Data center (1000s of users) |
|---|---|---|---|
| **llama.cpp** | ✅ ideal — portable, tiny footprint | ✅ good — `llama-server`, parallel slots | ❌ no multi-GPU serving story |
| **MLX** | ✅ ideal on Apple Silicon | ✅ good — `mlx_lm.server`, Mac mini economics | ❌ no rack-scale hardware path |
| **vLLM** | ⚠️ overkill — pre-allocates VRAM for absent users | ✅ strong with a CUDA GPU + real traffic | ✅ ideal — this is its home turf |

The pattern: **engines are shaped by their target concurrency.** llama.cpp optimizes the floor (run anywhere), vLLM optimizes the ceiling (serve everyone), and MLX optimizes a specific, increasingly interesting middle: serious models on hardware people already own.

### 6.3 Why MLX is where this series goes next

MLX is the youngest of the three, but it's the one whose constraint set looks most like the future of *personal* inference:

- **Unified memory changes the arithmetic.** The GPU-VRAM wall that defines llama.cpp offloading and vLLM paging simply isn't there; memory-per-dollar on a Mac mini/Studio is hard to beat for local models.
- **The ecosystem has matured fast**: the [mlx-community](https://huggingface.co/mlx-community) Hub org hosts thousands of pre-converted quantized models, `mlx_lm.server` speaks the OpenAI protocol with continuous batching, and `mx.distributed` runs inference across multiple Macs over Thunderbolt ([Apple's MLX project page](https://opensource.apple.com/projects/mlx/), [WWDC26 session](https://developer.apple.com/videos/play/wwdc2026/232/)).
- **It's a full framework, not just a runtime.** Unlike GGUF-consuming llama.cpp, MLX does training, LoRA fine-tuning, and quantization natively — the whole checkpoint-to-inference story from this document can happen inside one toolchain on one machine.

If the first essay in this series argued that access to inference is the new divide, MLX is the most concrete counter-argument available today: frontier-adjacent inference on a consumer device you own outright, with no meter running. The next document will go deep on it.

---

## Sources

- Hugging Face — [safetensors documentation](https://huggingface.co/docs/safetensors/index) · [safetensors GitHub (format spec)](https://github.com/safetensors/safetensors)
- DataCamp — [SafeTensors: secure ML model serialization](https://www.datacamp.com/blog/safetensors-format)
- ggml-org — [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) · [GGUF format deep dive (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/7.1-gguf-file-format) · [GGUF explained (APXML)](https://apxml.com/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gguf-format)
- vLLM — [project site](https://vllm.ai/) · [vLLM overview (Wikipedia, incl. SOSP 2023 PagedAttention paper)](https://en.wikipedia.org/wiki/VLLM) · [PagedAttention & continuous batching explained (Runpod)](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching)
- Apple — [MLX GitHub](https://github.com/ml-explore/mlx) · [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm) · [Apple Open Source: MLX](https://opensource.apple.com/projects/mlx/) · [WWDC25: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/) · [WWDC25: Explore LLMs on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/) · [WWDC26: Run local agentic AI on the Mac using MLX](https://developer.apple.com/videos/play/wwdc2026/232/)
- vLLM deployment tuning — [Optimization and Tuning (official docs)](https://docs.vllm.ai/en/stable/configuration/optimization/) · [gpu_memory_utilization tuning guide (DevOpsBeast)](https://devopsbeast.com/blog/vllm-gpu-memory-utilization)
- Transformer forward pass — [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) · [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
