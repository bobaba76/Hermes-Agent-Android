# Contributing to Hermes Agent

Thanks for your interest. Hermes is a privacy-first Android AI agent that aims to
stay useful with no cloud provider configured at all, and to grow onto the desktop
without rewriting the agent. All skill levels welcome — the issue tracker ranges
from self-contained bugs to multi-week features.

## Table of contents
- [Setting up](#setting-up)
- [The two-repo layout](#the-two-repo-layout)
- [How to contribute](#how-to-contribute)
- [Issue labels](#issue-labels)
- [PR guidelines](#pr-guidelines)
- [Architecture in 60 seconds](#architecture-in-60-seconds)
- [High-priority areas](#high-priority-areas)
- [Code style](#code-style)

---

## Setting up

**Requirements:** JDK 21 (JetBrains Runtime), a recent Android Studio, Android SDK
with **NDK 28.2** and CMake (the `llama.cpp` bridge is built from source),
`minSdk 29` / `targetSdk 36`.

```bash
# The engine is a separate repo and is NOT vendored — clone both, side by side.
git clone https://github.com/l3ad3r1/agent-core.git
git clone https://github.com/l3ad3r1/Hermes-Agent-Android.git

cd Hermes-Agent-Android
git submodule update --init      # pinned llama.cpp
./gradlew :app:assembleDebug     # ~90 MB debug APK, compiles native libs
./gradlew :app:installDebug
./gradlew test                   # unit tests
```

If `agent-core` is missing, configuration fails with an explicit error telling you
to clone it — that is expected, not a broken build.

**Getting a reply out of it.** The app builds and installs with no API key, but
there is no built-in mock: you need either a cloud provider (**Settings →
Assistant → Providers** — OpenAI, OpenRouter, Nous, Gemini, Groq, DeepSeek, or any
OpenAI-compatible `/v1` endpoint) or an on-device model downloaded from the in-app
catalogue. Keys are entered in-app and encrypted; nothing is baked into the build.

See [docs/BUILD.md](docs/BUILD.md) for the release toolchain and endpoint swaps.

---

## The two-repo layout

| Repo | Contents |
|---|---|
| **Hermes-Agent-Android** (this one) | Compose UI, app identity, signing, the `llama.cpp` JNI bridge (`app/src/main/cpp/ai_chat.cpp`), app-specific tool bindings. |
| **[agent-core](https://github.com/l3ad3r1/agent-core)** | The shared engine: domain contracts, model routing, tools, memory, persistence, settings. |

The split exists because two shipping apps run on one engine — Hermes and
**[Jeeves](https://github.com/l3ad3r1/Jeeves)** — so engine changes land in both
instead of drifting. It also keeps the engine free of Android-specific code, which
is what makes the desktop target a port rather than a rewrite.

**The one thing that will catch you.** This app pins the engine commit it builds
against in `agent-core.ref`, and **CI honours that pin while your local build
ignores it** (locally, `:core:*` maps straight onto your working tree). If your
change touches a shared API or JNI signature *and* its caller here, bump
`agent-core.ref` in the same PR — otherwise it compiles cleanly for you and fails
in CI on a signature nothing locally disagrees with. This app shipped v1.0.2 with
red CI for exactly that reason.

`ai_chat.cpp` is **byte-identical** with the copy in Jeeves. Change it in both, in
the same change; `md5sum` them before opening the PR.

---

## How to contribute

1. **Check existing issues first** — someone may already be on it.
2. **Bugs:** open an issue with `bug` before sending a PR. Include repro steps,
   expected vs actual, Android version and device.
3. **Features:** open an `enhancement` issue to agree the approach first. Large
   ones should reference [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
4. **Fork → branch → PR.** Branches: `fix/<short>` or `feat/<short>`, targeting `main`.

---

## Issue labels

| Label | Meaning |
|---|---|
| `bug` | Broken, or not working as documented |
| `good first issue` | Self-contained, roughly 1–2 hours, minimal context |
| `enhancement` | New capability in the existing app |
| `desktop` | Work toward the desktop target |
| `llm-backend` | On-device LLM, embeddings, retrieval |
| `plugin` | Plugin framework, sandbox, registry |
| `security` | Keystore, pinning, permissions |
| `ui` | Compose, theming, accessibility |
| `performance` | Memory pressure, startup, battery, inference speed |
| `test` | Missing or broken tests |
| `docs` | Documentation gaps |
| `help wanted` | A PR here would be especially welcome |

---

## PR guidelines

- One logical change per PR — easier to review, easier to revert.
- Every new class needs at least one unit test; see `app/src/test/` for patterns.
- Run `./gradlew test lintDebug` before pushing. Fix errors; warnings are advisory.
- No hardcoded API keys, credentials or device-specific paths.
- ProGuard: if you add a `@Keep` or `-keep` rule, say why in the PR description.
- Commit subject in the present tense, 72 chars or fewer ("Add X", not "Added X").
- Say whether an `agent-core.ref` repin is needed.

---

## Architecture in 60 seconds

```
UI (Compose) → ViewModel → Domain (interfaces) ← Data (implementations)
```

- **`domain/`** (in agent-core) — pure Kotlin, no Android imports. Models,
  repository interfaces, agent contracts.
- **`data/`** — Room, Retrofit, `LlmProvider` implementations, memory/RAG, plugins, tools.
- **`ui/`** — Compose screens and ViewModels. ViewModels talk to repositories;
  screens talk to ViewModels only.
- **`di/`** — Hilt modules; tools are multibound here.
- **`work/`** — WorkManager jobs (memory consolidation).

A turn goes `AgentRouter` → `OrchestratorImpl` → a per-step tool-call loop across
five roles. Deterministic phone commands are parsed locally and never reach a model.

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/MODULES.md](docs/MODULES.md) ·
[docs/LOCAL-INFERENCE-PERF.md](docs/LOCAL-INFERENCE-PERF.md) for the on-device engine
and the invariants that keep prefill fast.

---

## High-priority areas

### Desktop — Compose Multiplatform ([#7](https://github.com/l3ad3r1/Hermes-Agent-Android/issues/7))
The long-term goal, and the most impactful place to help. `agent-core` is already
platform-neutral, so this is largely extracting a shared UI layer and a desktop
entry point rather than reimplementing the agent.

### Ship the embedding model ([#3](https://github.com/l3ad3r1/Hermes-Agent-Android/issues/3))
`MiniLmEmbeddingService` is real and DI-bound — ONNX Runtime, all-MiniLM-L6-v2
int8, 384-dim, mean-pooled and L2-normalised. But it reads `model.onnx` and
`vocab.txt` from `AI Models/embeddings/all-MiniLM-L6-v2` on shared storage and
silently falls back to `HashingEmbeddingService` when they are absent, and nothing
downloads them. Wiring the model into the download catalogue would make good
retrieval the default rather than a lucky accident.

### Persistent vector store ([#4](https://github.com/l3ad3r1/Hermes-Agent-Android/issues/4))
`InMemoryVectorStore` loses every vector on process death and rebuilds by
re-embedding. sqlite-vec or SQLite-VSS behind the `VectorStore` interface.

### LLM-based fact extraction ([#10](https://github.com/l3ad3r1/Hermes-Agent-Android/issues/10))
Memory consolidation still uses a regex extractor.

### Certificate pinning ([#5](https://github.com/l3ad3r1/Hermes-Agent-Android/issues/5))
Not applied today because the endpoint is user-configurable. Opt-in pinning for
fixed-provider users. Self-contained; a good first security PR.

### gRPC plugin sandbox ([#6](https://github.com/l3ad3r1/Hermes-Agent-Android/issues/6))
`GrpcPluginSandbox` is an interface stub; real process isolation for third-party
plugins.

---

## Code style

Match the surrounding code. Kotlin official style, 4-space indent, explicit
visibility on public API. Comments should explain *why*, not restate the code —
the existing codebase leans on that heavily, especially around the inference and
approval paths where the non-obvious constraint is the whole point.
