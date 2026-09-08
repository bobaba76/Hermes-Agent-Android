# Local inference performance

How on-device chat got fast, and the invariants that keep it that way.

Landed 2026-09-07 in `259e2b6` (merged as `03eb4d1`, shipped in 1.0.2); §6
followed on 2026-09-08 and ships in 1.0.3. The
native side lives in `app/src/main/cpp/ai_chat.cpp`, which is **byte-identical
between this repo and Jeeves** — a change here belongs in both.

The short version: a conversation used to throw away its computed prefix on
every single turn, three separate ways. Each fix is small; together they are the
difference between a usable local model and one you stop using.

---

## 1. Prefix reuse in `processSystemPrompt`

**The bug.** `LocalLlmManager.generateResponse` rebuilds the whole system block
from scratch each call — that is deliberate on the Kotlin side, since the block
carries live state. The native side then *cleared the KV cache and re-prefilled
all of it*, every turn. On a long thread that is over a thousand tokens of
recompute to say one sentence.

**The fix.** Each lane keeps `cached_tokens`, a mirror of what is actually
resident in its sequence. Before prefilling, the new system block is compared
against it:

```cpp
const int n_reused = (int) common_prefix_length(lane.cached_tokens, system_tokens);
llama_memory_seq_rm(llama_get_memory(g_context), lane.id, n_reused, -1);
lane.cached_tokens.resize(n_reused);
lane.current_position = n_reused;

const llama_tokens pending(system_tokens.begin() + n_reused, system_tokens.end());
```

Everything up to the point of divergence is already computed; only the tail is
decoded. A turn that appends to a thread reuses nearly all of it. A different
conversation shares no prefix and correctly falls back to a full prefill.

`common_prefix_length` is a plain token-by-token walk (`ai_chat.cpp:456`) — the
mirror is what makes it possible, not the comparison.

## 2. `n_ubatch` was throttling every prefill 8×

`n_ubatch` was pinned to **64** as a workaround for `vk::DeviceLostError` (TDR)
on Adreno. Vulkan is compiled out (`-DGGML_VULKAN=OFF`) and every decode runs on
the CPU, so the cap survived its own reason and was splitting each 512-token
prefill chunk into eight micro-batches for nothing.

Now `n_ubatch = BATCH_SIZE` (512), matching `n_batch`.

> If Vulkan is ever switched back on, this is the line to revisit **before**
> blaming the GPU backend.

## 3. `n_gpu_layers` follows the registered backends

It was a hardcoded `0`. Now:

```cpp
model_params.n_gpu_layers = has_gpu_backend() ? GPU_OFFLOAD_LAYERS : 0;
```

A CPU-only build behaves exactly as before, and enabling a GPU backend does not
require a second, easily-forgotten code change.

## 4. KV lanes — background inference can no longer evict the chat

**The bug.** One llama.cpp sequence served everything. Any auxiliary inference —
in practice the conversation-brief merge — shares only ~5 tokens of prefix with
the chat prompt, so running one **wiped the chat's KV cache**. Every chat turn
after a summarisation started cold, which is the worst possible pairing: the
expensive thing triggers the other expensive thing.

**The fix.** Two lanes, each a llama.cpp sequence id:

```cpp
constexpr int N_LANES           = 2;
constexpr int LANE_CHAT         = 0;
constexpr int LANE_AUX          = 1;
constexpr int LANE_CONTEXT_SIZE = DEFAULT_CONTEXT_SIZE / N_LANES;   // 8192 / 2
```

All per-conversation state (cached-token mirror, `chat_msgs`, positions, sampler,
assistant buffer) moved into a `Lane` struct; `g_lanes[N_LANES]` holds them.

The part worth understanding is that **this costs no extra KV memory**:

```cpp
ctx_params.n_seq_max  = n_seq_max;   // N_LANES
ctx_params.kv_unified = false;
```

With `kv_unified = false`, llama.cpp partitions the existing `n_ctx` rather than
allocating a second cache — `n_ctx_seq = n_ctx / n_seq_max`. The 8192-token
context becomes 4096 per lane. Nothing grows; it is divided.

> ### Invariant
> **Anything reasoning about how much room a conversation has must use
> `LANE_CONTEXT_SIZE`, never `DEFAULT_CONTEXT_SIZE`.** Overflow and truncation
> checks already do (`ai_chat.cpp:478`, `:544`). Using the total silently
> over-commits by 2× and ends in a mid-generation failure.
>
> `benchModel` deliberately builds a single-sequence context
> (`init_context(g_model, pp, /* n_seq_max */ 1)`) so a benchmark measures the
> whole context, not half of it.

## 5. The conversation brief was re-summarised every turn

Lives in `agent-core`:
`core/llm/src/main/kotlin/com/hermes/agent/data/llm/ConversationCompressor.kt`
(shared with Jeeves).

**The bug.** `ChatRepositoryImpl` splits a thread into `recent` (verbatim) and
`older`, and asked for a brief of `older` keyed on an **anchor id** — the id of
the newest *older* message. On a growing thread that id advances every turn, so
the cache could never hit and the entire history was re-summarised each time.

Compounding it: that summarisation ran on the shared sequence (§4), so it also
evicted the chat prefix and forced the next chat prefill to start cold.

**The fix.** The anchor is gone; `brief()` now takes only the conversation id and
`older`, and maintains the summary **incrementally**:

- A cached brief records `coveredCount` — how many of `older` it accounts for.
- Turns that dropped out of the verbatim window since are carried **verbatim**
  as a tail alongside the brief. Nothing is forgotten between merges, and it
  costs no inference.
- The tail is folded in only when it is worth an inference:
  `MERGE_AFTER_MESSAGES = 6` turns or `TAIL_CHARS = 2000` characters. The merge
  summarises *brief + tail*, never the whole thread again, so cost stays bounded
  however long the conversation grows.
- A cached brief claiming to cover more than `older` contains belongs to a
  thread since trimmed or cleared; it cannot be reconciled, so it is discarded
  and rebuilt.

## 6. Model slots — a tool turn no longer evicts the chat model

§4 stopped background inference evicting the chat *cache*. This is the same
problem one level up: eviction of the chat **model**.

**The bug.** There was one `llama_model`/`llama_context` pair. The on-device tool
caller is a different model (FunctionGemma 270M) from the chat model, so a tool
turn unloaded the chat GGUF, loaded its own, and the next chat turn reloaded the
chat GGUF and re-prefilled from nothing. Lanes could not help — a lane divides
one context, and this needs two.

**The fix.** A `Slot` owns everything that was global, and each role gets one:

```cpp
struct Slot {
    llama_model             * model = nullptr;
    llama_context           * context = nullptr;
    llama_batch               batch{};
    common_chat_templates_ptr chat_templates;
    Lane                      lanes[N_LANES];
};
static Slot g_slots[N_SLOTS];          // SLOT_CHAT = 0, SLOT_TOOL_CALLER = 1
```

Lanes nest inside slots, so the two ideas compose: each model keeps its own chat
and auxiliary sequences.

Unlike lanes, this **does** cost memory — two sets of weights and two KV caches —
so it is gated on headroom (`LocalLlmManager.canHoldBothModels`). A low-memory
device keeps the old one-slot swap. Weights are mmap'd, so the resident cost is
largely file-backed and reclaimable.

> ### Invariant
> **A setting that invalidates both models must say so.** `updateModelSelection`
> unloads the chat engine; the tool caller has its own slot and is untouched
> unless the caller passes `alsoToolCaller`. Only `setModelDownloadDir` does,
> because both GGUFs live in that one folder — the other settings just pick a
> chat model, and unloading the tool caller for those would buy a reload on the
> next tool turn for nothing.
>
> Nothing fails loudly if this is wrong: the tool caller simply goes on serving a
> model loaded from a folder the user has moved. `ToolCallerSlotCleanupTest`
> pins it.

---

## Measured

Samsung S24U (Snapdragon 8 Gen 3), Qwen2.5 1.5B Q4_K_M, long thread — taken
while making the change:

| | Before | After |
|---|---|---|
| Chat prefill | 1531 tokens, ~24 s (cold every turn) | reuses the prefix; only the divergent tail is decoded |
| Brief | ~38 s, **every turn** | once per ~6 turns |
| Turn latency | 60 s+ | — |
| Chat prefix across a tool turn (§6) | 0 reused, GGUF reloaded | 741/778 reused, zero reloads |

Prefill improved roughly **9×** on that thread. These numbers are from that
session's device runs and are not reproduced by any check in the repo; treat
them as the scale of the win, not a regression baseline.

## 7. OpenCL on Adreno — wired, off, and unverified

Opt-in via the `OPENCL_SDK` environment variable; see **`docs/BUILD.md` §4a**
for the build recipe. Not enabled in any shipped build.

Why OpenCL rather than Vulkan: it is the backend Qualcomm targets at Adreno,
upstream llama.cpp lists **Adreno 750 (Snapdragon 8 Gen 3) as verified** — the
S24U exactly — and it supports **Q4_K**, so the existing Q4_K_M catalogue works
unchanged. Do **not** re-quantise to Q4_0 for this path.

Two details that are easy to get wrong:

- The `libOpenCL.so` under `$OPENCL_SDK/lib/arm64-v8a/` is a **link-time stub
  only** and is deliberately not packaged into the APK. On device the loader
  resolves the soname to the vendor's `/vendor/lib64/libOpenCL.so` — the real
  Adreno driver — which is exported to apps via `/vendor/etc/public.libraries.txt`.
  Confirm that entry exists on the target before assuming this resolves.
- Building it in is safe on non-Adreno hardware (the backend simply does not
  register), but **it has never been built or run**. Vulkan died by losing the
  device mid-inference; assume nothing until this has run on real Adreno.

---

## Keeping it working

- `ai_chat.cpp` is shared byte-for-byte with Jeeves. Port changes both ways.
- Auxiliary inference must run on `LANE_AUX`. The native side does not decide
  this: the JNI entry points take a `jint jlane`, and `LocalLlmManager` picks it
  from a coroutine-context marker —

  ```kotlin
  val lane = if (currentCoroutineContext()[AuxiliaryInference.Key] != null)
      InferenceEngine.Lane.AUXILIARY else InferenceEngine.Lane.CHAT
  ```

  so **any background inference must be launched inside
  `withContext(AuxiliaryInference)`** (`LocalLlmManager.kt:59`, `:400`). Forget
  it and the work lands on `LANE_CHAT`, silently restoring the original bug: still
  correct, just slow again — exactly the kind of regression nothing fails on.
  `AuxiliaryInferenceContextTest` covers the marker.
- Use `LANE_CONTEXT_SIZE` for any per-conversation capacity maths.
- Lanes divide one model's context; slots are whole models. A new *role* needs a
  slot; a new *kind of inference for an existing role* needs a lane.
- The prefill and load log lines name the slot (`slot N lane M ...`). Without
  that a tool-caller prefill reads exactly like a chat one, which is what hid
  §6. When measuring, remember that touching any model setting unloads the chat
  model — a cold read straight after a settings change proves nothing.
- If prefill goes cold again, check in this order: is the Kotlin system block
  newly non-deterministic (a timestamp, a reordered set) so the prefix diverges
  at token 0; is something running on the wrong lane; has `n_ubatch` been pinned
  again.
