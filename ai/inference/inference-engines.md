# Inference Engines

Inference engines are software that help deploy trained pretrained tranformer models for inference i.e. send query get intelligent responses

## Similarities to Internet

## **Knowledge is Power** The old equation (early 2000s)

- The internet's promise: put the world's knowledge within reach of anyone with a connection
- Google's founding mission: *"organize the world's information and make it universally accessible and useful"*
- The fight of that era was the **digital divide** — getting people *connected*
- Once you were online, the knowledge itself was essentially **free**

## Knowledge + Inference is Power

- Knowledge is now abundant — the internet solved that
- What's scarce is **inference**: the compute-hungry act of *applying* a model to your problem
- Raw knowledge no longer differentiates; the ability to **reason over it at scale** does
- Every AI answer, every agent step, every generated line of code is a **metered act of inference**

## Imagine Google charged **$100/month** to search the internet.

- Homework happens only in households that can afford it
- The self-taught programmer, learning a new language, the patient researching a diagnosis — **priced out**
- Wikipedia, Stack Overflow, open-source — most of it never forms, because the audience that built it couldn't afford to browse

### The inference divide is already measurable

IMF — [AI Will Transform the Global Economy. Let's Make Sure It Benefits Humanity (2024)](https://www.imf.org/en/blogs/articles/2024/01/14/ai-will-transform-the-global-economy-lets-make-sure-it-benefits-humanity)


# The moment after training

Pretraining, fine-tuning, and RLHF differ enormously in *how* they update a model — but they all produce the **same kind of artifact**. When the last optimizer step runs, what exists is:

- **Weights** — hundreds of named tensors (a "state dict"): embedding matrices, attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), MLP layers, norms, and the output head. For a 7B-parameter model in 16-bit floats, that's ~14 GB of raw numbers.
- **Architecture config** — a small file (`config.json`) declaring the shape those tensors plug into: number of layers, hidden size, attention head counts, vocabulary size, RoPE parameters, context length.
- **Tokenizer** — the vocabulary and merge rules (`tokenizer.json`, `tokenizer_config.json`) that map text ↔ integer IDs. The model never sees text; it only sees these IDs.
- **Chat template** — for instruction/RLHF-tuned models, a Jinja template describing how to wrap a conversation in the special tokens the model was trained on (e.g. `<|im_start|>user ... <|im_end|>`). Get this wrong and a perfectly good model behaves badly.

Crucially, **weights alone are not a runnable thing**. They're inert data. An inference engine is the program that reconstructs the computation graph from the config, fills it with the weights, and executes it.

> Training artifacts vs. distribution artifacts: labs also checkpoint optimizer state (Adam moments, LR schedules) during training, which can triple the size. That's stripped before distribution — you ship only what inference needs.

## Three inference engines

| | llama.cpp | vLLM | MLX (mlx-lm) |
|---|---|---|---|
| Format consumed | GGUF | Hub safetensors | MLX-converted safetensors |
| Load strategy | mmap, lazy page-in, optional GPU offload | eager copy to VRAM, paged KV pre-allocation | load into unified memory, lazy graph |
| Killer feature | runs quantized models anywhere, zero deps | PagedAttention + continuous batching for many users | zero-copy CPU/GPU on Apple Silicon |
| Sweet spot | edge, laptops, single-file portability | multi-user GPU serving at scale | local dev on Macs |
| Text input | `llama-cli` / OpenAI-compatible `llama-server` / C API | `LLM.generate()` / OpenAI-compatible `vllm serve` | Python API / `mlx_lm.generate` / `mlx_lm.server` |

Transformer forward pass — [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) · [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

| | Local (1 user) | Small service (~10s of users) | Data center (1000s of users) |
|---|---|---|---|
| **llama.cpp** | ✅ ideal — portable, tiny footprint | ✅ good — `llama-server`, parallel slots | ❌ no multi-GPU serving story |
| **MLX** | ✅ ideal on Apple Silicon | ✅ good — `mlx_lm.server`, Mac mini economics | ❌ no rack-scale hardware path |
| **vLLM** | ⚠️ overkill — pre-allocates VRAM for absent users | ✅ strong with a CUDA GPU + real traffic | ✅ ideal — this is its home turf |

# MLX Hands-On: From Empty Directory to Served Tokens

## 1. Create the environment

MLX is a Python package (with C++/Metal underneath), so it lives in a virtual environment like any other. Two equivalent paths:

```bash
# Classic venv
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm

# Or with uv (faster, no activate ceremony needed)
uv venv
uv pip install mlx-lm
```

`mlx-lm` pulls in `mlx` (the array framework) plus the Hugging Face `transformers` tokenizer stack. That's the whole install — no CUDA toolkit, no driver matching, no compilation. This is one of MLX's quiet advantages: the "GPU driver" is macOS itself.

Sanity check:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# Device(gpu, 0)
```

## 2. Start the server

```bash
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080
```

First run downloads the model from the [mlx-community](https://huggingface.co/mlx-community) Hub org (pre-converted, pre-quantized safetensors — see §2.4 of the previous doc) into `~/.cache/huggingface/`. Subsequent starts load straight from disk into unified memory.

You now have an **OpenAI-compatible HTTP server** on `localhost:8080`. The endpoints that matter:

- `GET /v1/models` — list what's loaded
- `POST /v1/chat/completions` — chat (applies the chat template for you)
- `POST /v1/completions` — raw completion (no template — you're on your own)

> **A note on exposure:** `mlx_lm.server` speaks plain HTTP and performs no authentication. It is fine on `localhost`; it must never face the internet naked. Production deployments put a reverse proxy (nginx, Caddy, Traefik) in front to terminate TLS — which is exactly what the demo server below does, one step further: it requires **mutual TLS**.

## 3. Three requests, three responses

All responses below are **live captures** from `edai.ed-yahska.xyz` (long floats in `timings` rounded, one long answer abridged). All three examples hit the same endpoint — `POST /v1/chat/completions` — with the same envelope. What changes is what you put in the body: plain messages, a reasoning toggle, or a tool inventory.

### 3.1 A simple query

**Request:**

```bash
curl https://edai.ed-yahska.xyz/v1/chat/completions \
  --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [
      {"role": "user", "content": "What is inference, in one sentence?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-ae67246c-991d-4ef5-9796-334fd81b8523",
  "object": "chat.completion",
  "created": 1784140344,
  "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Inference is the process of drawing logical conclusions or making educated guesses based on available evidence, observations, or reasoning rather than explicit statements.",
        "reasoning_content": null,
        "tool_calls": null
      }
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "completion_tokens": 28,
    "total_tokens": 49,
    "prompt_tokens_details": {"cached_tokens": 0}
  },
  "timings": {
    "prompt_n": 21, "predicted_n": 28,
    "prompt_per_second": 2.5, "predicted_per_second": 38.8,
    "peak_memory": 15.42
  }
}
```

(A charming detail: asked cold, the model gave the *dictionary* definition of inference, not the ML one. Context matters.)

Reading the response: `choices[0].message` is the assistant turn you append to the conversation for the next request (the server is stateless — *you* keep the history); `finish_reason: "stop"` means the model emitted its end-of-turn token rather than hitting `max_tokens` (which would read `"length"`); `usage` is the token accounting — prompt tokens are the prefill, completion tokens are the decode loop.

This server also returns two useful extensions beyond the standard OpenAI schema. `timings` makes the prefill/decode split from the previous document *visible*: decode runs at a steady ~37–38 tokens/sec across every request in this section, while prefill throughput swings wildly (2.5 tokens/sec here on a cold model, 74–120 tokens/sec once warm) — and `peak_memory` (GB) shows what the loaded weights plus KV cache actually occupy in unified memory. And `prompt_tokens_details.cached_tokens` exposes prompt caching: replayed conversation prefixes can skip prefill entirely.

### 3.2 Reasoning enabled

Reasoning models (Qwen3, DeepSeek-R1 distills, etc.) are trained to emit a *thinking block* before the answer, and OpenAI-compatible servers expose it either as `<think>...</think>` tags inside `content` or in a dedicated `reasoning_content` field. The request-side switch is a chat-template argument:

**Request:**

```bash
curl https://edai.ed-yahska.xyz/v1/chat/completions \
  --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [
      {"role": "user", "content": "I have 3 GGUF files of 4.2 GB each and 16 GB of RAM. Can I mmap all three at once?"}
    ],
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 1024,
    "temperature": 0.6
  }'
```

**Response (abridged — the full answer ran 860 tokens):**

```json
{
  "id": "chatcmpl-cd531ec1-91ce-4e0b-b72c-75f9b557fb00",
  "object": "chat.completion",
  "created": 1784140381,
  "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "The short answer is **yes, you can technically mmap them, but whether they will run smoothly depends entirely on how you are using them.**\n\n### 1. The \"Magic\" of mmap\nMemory mapping (`mmap`) does not actually load the files into your RAM immediately. Instead, it tells the operating system: *\"Map these files to the virtual address space. Only pull the data from the disk into physical RAM when the CPU actually tries to read it.\"*\n\nBecause you have 16 GB of RAM and $4.2 \\times 3 = 12.6$ GB of data, you are under the 16 GB limit. [...]\n\n**B. The RAM Overhead (Stability)**\nWhile the files are 12.6 GB, the **context window** (the KV Cache) is not part of the GGUF file; it is generated in your RAM as you chat. [...] Once you cross the 16 GB threshold, your OS will start using the **Swap File** [...]",
        "reasoning_content": null,
        "tool_calls": null
      }
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 860,
    "total_tokens": 905,
    "prompt_tokens_details": {"cached_tokens": 0}
  },
  "timings": {
    "prompt_n": 45, "predicted_n": 860,
    "prompt_per_second": 74.0, "predicted_per_second": 36.8,
    "peak_memory": 15.61
  }
}
```

The interesting part of this capture is what **didn't** happen: `reasoning_content` is `null`. The server accepted `enable_thinking` without complaint, and the answer is thorough — but there is no separate reasoning channel, because **reasoning is a property of the model, not a server switch**. Neither model on this deployment (gemma-4-it, Qwen2.5-Instruct) was trained to emit thinking tokens, so the template argument is a no-op. This is the "chat template is part of the model contract" lesson from the previous document, seen from the API side: the field exists in the schema; only a reasoning-trained model fills it.

Swap in a reasoning model (say, an mlx-community Qwen3 or R1-distill quant) and the *same request* returns the thinking separated out:

```json
// Illustrative — what a reasoning-trained model returns for the same request
"message": {
  "role": "assistant",
  "reasoning_content": "Three files × 4.2 GB = 12.6 GB against 16 GB RAM. mmap maps address space, not resident memory — pages fault in on access and evict under pressure since the file is the backing store. So the mapping succeeds; the real question is the active working set...",
  "content": "Yes — mmap reserves address space, not RAM, so mapping all three works. The real constraint is the active working set: ..."
}
```

Note the cost either way: this answer consumed **860 completion tokens vs. 28** for example 4.1 — a ~30× difference in decode time for one question. Thoroughness (and, with reasoning models, thinking) is a dial on the quality-vs-cost tradeoff; that's why the toggle exists at all.

### 3.3 Tool calling

Tool calling is a structured conversation: you advertise functions in the request; if the model decides one is needed, it responds with a `tool_calls` message instead of prose; you execute the function and send the result back as a `role: "tool"` message; the model then answers in natural language.

**Request (turn 1 — advertise the tool):**

```bash
curl https://edai.ed-yahska.xyz/v1/chat/completions \
  --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [
      {"role": "user", "content": "How hot is it in San Jose right now?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City name"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "max_tokens": 256
  }'
```

**Response (turn 1 — the model asks you to run the tool):**

```json
{
  "id": "chatcmpl-9d107e49-3655-4ccd-87fc-f5a4003b4774",
  "object": "chat.completion",
  "created": 1784140503,
  "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "choices": [
    {
      "index": 0,
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "index": 0,
            "id": "3cef584a-44b6-4833-bc2e-04f127136e67",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"city\": \"San Jose\"}"
            }
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 101,
    "completion_tokens": 18,
    "total_tokens": 119,
    "prompt_tokens_details": {"cached_tokens": 0}
  },
  "timings": {
    "prompt_n": 101, "predicted_n": 18,
    "prompt_per_second": 6.0, "predicted_per_second": 37.2,
    "peak_memory": 30.58
  }
}
```

Four things to notice in this capture:

- `finish_reason: "tool_calls"` and `content: null` — the model produced no prose, only a structured request for you to act.
- `arguments` is a **JSON string, not a JSON object** — the model literally generated those characters token by token (the chat template taught it the format; the server parsed them out of the raw output).
- The model passed only the *required* argument (`city`) and skipped the optional `unit` — its judgment call, and exactly the kind of behavior tool descriptions exist to steer.
- `prompt_tokens` jumped to 101, from 21 for the same-length user question in example 4.1 — **the tool schemas are injected into the prompt**, so every advertised tool costs context on every request.

**Request (turn 2 — return the tool result):**

```bash
curl https://edai.ed-yahska.xyz/v1/chat/completions \
  --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [
      {"role": "user", "content": "How hot is it in San Jose right now?"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "3cef584a-44b6-4833-bc2e-04f127136e67", "type": "function",
         "function": {"name": "get_current_weather",
                      "arguments": "{\"city\": \"San Jose\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "3cef584a-44b6-4833-bc2e-04f127136e67",
       "content": "{\"temp\": 84, \"unit\": \"fahrenheit\", \"conditions\": \"sunny\"}"}
    ],
    "max_tokens": 128
  }'
```

**Response (turn 2 — natural-language answer):**

```json
{
  "id": "chatcmpl-7d39373b-71b2-4982-929e-23c6d95a4c33",
  "object": "chat.completion",
  "created": 1784140517,
  "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "It is currently 84°F in San Jose with sunny conditions."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 71,
    "completion_tokens": 16,
    "total_tokens": 87,
    "prompt_tokens_details": {"cached_tokens": 0}
  },
  "timings": {
    "prompt_n": 71, "predicted_n": 20,
    "prompt_per_second": 119.8, "predicted_per_second": 38.4,
    "peak_memory": 30.58
  }
}
```

This two-turn loop *is* the agent loop, at its smallest. Everything an agentic framework does is this exchange, repeated, with more tools.

### 3.4 LLMs are stateless

Every example above said "the client owns the conversation." Here's the proof in three requests (live captures, abridged). Tell the model your name:

```bash
curl -sS https://edai.ed-yahska.xyz/v1/chat/completions --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-26b-a4b-it-4bit",
       "messages": [{"role": "user", "content": "Hi! My name is Akshay and I write a blog about AI inference."}],
       "max_tokens": 64}'
# → "Hi Akshay! It's great to meet you. ..."            (prompt_tokens: 28)
```

Ask for it back in a fresh request — seconds later, same server, same model:

```bash
curl -sS https://edai.ed-yahska.xyz/v1/chat/completions --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-26b-a4b-it-4bit",
       "messages": [{"role": "user", "content": "What is my name?"}],
       "max_tokens": 64}'

# → "I do not know your name. As an AI, I only have access to the information
#    provided during our current conversation, and you haven't told me your
#    name yet."                                          (prompt_tokens: 18)
```

Total amnesia — it even insists you never introduced yourself. Now the same question with the history replayed in the messages array:

```bash
curl -sS https://edai.ed-yahska.xyz/v1/chat/completions --cert demo-bundle.pem \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-26b-a4b-it-4bit",
       "messages": [
         {"role": "user",      "content": "Hi! My name is Akshay and I write a blog about AI inference."},
         {"role": "assistant", "content": "Hi Akshay! It'\''s great to meet you."},
         {"role": "user",      "content": "What is my name?"}
       ],
       "max_tokens": 64}'
```
---

### 3.5. What these three examples add up to

- **One endpoint, one envelope.** Simple chat, reasoning, and tool use are all `POST /v1/chat/completions` with a `messages` array — the differences are a template kwarg and a `tools` list. This is why the OpenAI-compatible surface won: a Mac mini running `mlx_lm.server` behind mTLS is a drop-in substitute for a cloud API.
- **The server is stateless; the client owns the conversation.** Each request replays the full history. (This is also why prompt/KV caching matters so much for multi-turn cost.)
- **Everything is tokens.** Reasoning shows up as generated tokens (in `content` or split into `reasoning_content`); tool calls are tokens parsed into structure; tool schemas are tokens charged to your prompt. The `usage` and `timings` blocks in each response are the meter — and on your own hardware, it's a meter with no bill attached.

---

## 4 Demos

Let's start with the basics. The bundle holds both the client cert and its EC key, so a single --cert covers both — no separate --key needed:

```bash
# List models
curl -sS --cert secrets/demo-bundle.pem https://edai.ed-yahska.xyz/v1/models
```

```bash
# Server health (shows which model is currently loaded)
curl -sS --cert secrets/demo-bundle.pem https://edai.ed-yahska.xyz/health
```

Now the endpoint everything else is built on — this is the call an agent actually makes:

```bash
curl -sS --cert secrets/demo-bundle.pem \
  https://edai.ed-yahska.xyz/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "Say pong"}],
    "max_tokens": 64
  }'
```

Next, streaming. Add stream: true, and -N to disable curl's buffering, and the response arrives as data: SSE chunks carrying delta.content:

```bash
curl -sSN --cert secrets/demo-bundle.pem \
  https://edai.ed-yahska.xyz/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "Count 1 to 3"}],
    "max_tokens": 64,
    "stream": true
  }'
```

And finally tool calling — the shape the whole agent loop rests on. Watch for finish_reason: "tool_calls" and a populated tool_calls array in the response:

```bash
curl -sS --cert secrets/demo-bundle.pem \
  https://edai.ed-yahska.xyz/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "messages": [{"role": "user", "content": "Run `uname -s` for me"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "run_command",
        "description": "Run a shell command.",
        "parameters": {
          "type": "object",
          "properties": {"cmd": {"type": "string"}},
          "required": ["cmd"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 2048
  }'
```

## 5 Power profiles

Here's the part that surprised me. The Mac mini serving every request in this document stays powered on around the clock — and a full day of that costs **one cent**. $0.01.

The meter tells the story: **0.090 kWh** across the day, averaging **4 watts**. Four watts is a phone charger. That's the floor the machine idles at between requests, and idling is what a personal inference server does almost all of the time.

Now the interesting number. That tall stretch in the evening looks like a spike only because this is a day view — it's a **full hour of sustained inference at 35 watts**, the machine working as hard as this workload asks it to. So take that hour and run it out: 24 of them back to back is **0.84 kWh**. At 15–17¢ per kilowatt-hour, that's **about 13 to 14 cents a day.**

That's the ceiling, not the average. A Mac mini pinned at full tilt, serving tokens every second of every day, for the price of a stick of gum. Nine times the energy of the real day above — and still under a quarter.

This is the meter with no bill attached, made literal.

<img width="320" alt="cost" src="https://raw.githubusercontent.com/ed-yahska-xyz/blogs/main/ai/inference/assets/cost.jpg">

<img width="320" alt="kwh" src="https://raw.githubusercontent.com/ed-yahska-xyz/blogs/main/ai/inference/assets/kwh.png">
