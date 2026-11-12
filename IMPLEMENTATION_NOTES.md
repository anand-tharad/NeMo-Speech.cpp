# Implementation notes — Hindi streaming ASR service

Working log for building a streaming ASR service on NeMo-Speech.cpp. Newest entries at the bottom
of each section. Full Phase 0 reconnaissance lives in the approved plan; this file records what was
done, run, and learned.

## Target

Hindi streaming ASR. Prompt-conditioned FastConformer-Transducer fine-tuned on Hindi data with the
`hi-IN` prompt tag. Goal: mic session with partials while speaking, finals at utterance end,
continuous multi-utterance session, explicit finish.

## Environment

| Where | Role |
|---|---|
| Windows workspace | Source reading and code authoring only. No build toolchain (no cmake/ninja/cl/nvcc), submodules uninitialized, torch is CPU-only, no model artifacts. |
| Linux server (`edgexpert-d029`) | All builds, conversions, and runs. CUDA available (`backend=CUDA0`). |

Consequence: no command is reported as successful here unless its output was pasted back from the
server.

## Architecture discoveries (Phase 0)

The repository already implements most of what the original brief proposed to build:

- **Streaming session API** — `asr::RecognitionStream` (`src/asr/recognizer.h:67`):
  `push()` / `force_endpoint()` / `next()` / `finish()`. Mirrored in the stable C ABI
  (`include/nemo_speech/asr.h:213-241`). Both network transports are thin adapters over it.
- **Live mic client** — `transcribe_live()` (`app/transcribe.cpp:356`), miniaudio capture at
  16 kHz mono f32, partial/final split already separated in `drain_results` (`:438`).
- **Endpointing** — `asr::VadEndpointer` (`src/asr/vad/vad_endpointer.h:34`) with Silero VAD and a
  feature masker; `--endpointing` is **on by default with `--live`**, `--stop-history-eou-ms`
  defaults to 800.
- **Config** — YAML via `--config`, dotted overrides via `--asr.SECTION.KEY`; precedence is
  defaults < YAML < `NEMO_SPEECH_*` env < CLI.
- **Servers** — WebSocket realtime transcription and Riva gRPC `StreamingRecognize`, both gated
  OFF by default at build time.

Key mechanics worth remembering:

- **Chunk geometry** is `steady = 9 + 8*(1+R)` mel frames, advancing `shift = 8*(1+R)`. A chunk
  always needs more frames than it consumes, so a lookahead remains unprocessed until finalization.
- **`rnnt_right_context` defaults to `1`, not `-1`** (`src/asr/runner.h:34`) — the runtime silently
  overrides the model's trained R. **Must pass `3` explicitly for this model.**
- **Ctrl+C is a graceful finalize**, not a kill: the flag stops the loop, the mic tail is drained
  and pushed, then `finish()` runs (`app/transcribe.cpp:488-494`). No audio is lost.
- **Mid-stream EOU preserves acoustic state** — `fire_eou()` (`src/asr/runner.cpp:95-118`) clears
  per-utterance text but deliberately keeps encoder caches, predictor state, mel overlap, and the
  stream clock. Continuous multi-utterance sessions already work correctly.
- **Backpressure** is a bounded 2 s lock-free ring that **drops newest on overflow**, silently
  (`app/microphone_capture.cpp:28,46-54`).
- **Streaming feeds raw un-normalized log-mel** by design (`src/asr/runner.cpp:990-995`).

## Commands executed (on the server)

### Conversion

```bash
python convert_model.py .../FastConformer-Transducer-BPE-Prompt-Streaming-new.nemo \
  -o hindi-q8.gguf --architecture asr --head-type rnnt
```

Result: `hindi-q8.gguf`, 812.9 MB. Converter log reported:

| Fact | Value |
|---|---|
| `head_type` | `rnnt` |
| `cache_supported` | `True` |
| `train_left_ctx` / `train_right_ctx` | **56 / 3** — `[56, 3]` survived as `att_context_size` pair[0] |
| `num_prompts` / languages | 128 / 121 |
| `feat_in` | **128** mels (not the 80 struct default) |
| `vocab_size` / `blank_id` | 13088 / 13087 |
| `subsample` | 8 |
| dtype tally | F16=78, F32=353, Q8_0=224 |

Two tensors skipped (`preprocessor.featurizer.{window,fb}`) — **benign**: they are in
`SKIP_PREFIXES` because the filterbank is regenerated via librosa and written as `preprocessor.fb`
(`conversion/asr.py:675`), which the runtime reads (`src/asr/model.cpp:1328`).

`--metadata-json` was passed and silently produced no file. It is declared
(`convert_model.py:56`) and forwarded into `ConversionRequest` (`:104`), but the ASR branch of
`conversion/registry.py` never passes it on — only `tts`, `codec`, and `s2s` honor it. Accepted and
dropped without warning.

### GGUF validation

```bash
nemo-speech model info hindi-q8.gguf
```

`runtime_compatible: true`, no errors, no warnings, `tensor_count: 657`. The 2-tensor difference
from the converter's "emitted 655" is the regenerated filterbank and window, confirming
`preprocessor.fb` is present. Prompt dictionary includes `hi-IN` (and `mr-IN`, `bn-IN`, etc.);
`hi-IN` matches the fine-tuning, so it is the correct and only tag for these weights.

### Streaming transcription

```bash
nemo-speech transcribe --model hindi-q8.gguf --stream \
  --asr.streaming.rnnt_right_context 3 -l hi-IN <clip>.wav
```

Runtime status line:

```
[asr] model=hindi-q8 head=rnnt backend=CUDA0
[asr] mode=streaming head=rnnt left=56 center=1 right=3 attention=60 encoder-frames step=320ms
```

`attention=60` = 56+1+3 and `step=320ms` = `8*(1+3)` mel frames — the derived geometry is confirmed
by the runtime itself, and the trained mode is active rather than the R=1 default.

Two clips produced complete, readable Hindi transcripts.

## Test results

| What | Result |
|---|---|
| Conversion | Pass — valid GGUF, geometry and prompts preserved |
| GGUF load / compatibility check | Pass — `runtime_compatible: true` |
| Streaming decode, 2 short clips (CUDA) | Pass by inspection — complete transcripts, no padding, no workaround |
| Live mic session (Windows, CPU backend) | **Pass — all three acceptance checks met** |
| WER / CER | **Not measured** — no reference transcripts yet |

### Live mic session — 31.4 s, four utterances, Windows CPU backend

```
nemo-speech.exe transcribe --live --model hindi-q8.gguf \
  --asr.streaming.rnnt_right_context 3 -l hi-IN --stop-history-eou-ms 800
```

```
[asr] model=hindi-q8 head=rnnt backend=CPU
[asr] mode=streaming head=rnnt left=56 center=1 right=3 attention=60 encoder-frames step=320ms
  LIVE  Microphone Array (AMD Audio Device)

  Final  0:03.2 - 0:04.6   मुझे चार दिन से बुखार है
  Final  0:09.6 - 0:11.0   एंड पॉइंटिंग चेक कर रहा हूँ
  Final  0:15.4 - 0:15.7   चेक
  Final  0:17.3 - 0:17.9   एंड क्यों नहीं हो रहा है
  Stopped after 31.4s of captured audio
```

Acceptance criteria, all met with **no code changes**:

1. Partials update while speaking — yes.
2. A separate final lands after each pause — yes, four of them, endpointing firing on the
   800 ms token-silence threshold.
3. Session survives multiple utterances without restart — yes, encoder and predictor state
   preserved across all four EOUs as designed.

Ctrl+C emitted **no fifth final**, which is correct: all four utterances had already been closed by
EOU, so `finish()` had no pending audio and returned an empty final that `commit_final` correctly
did not print. The concatenated transcript printed once at the end, without re-joining partials.

## Findings

**The missing-final-words problem did not reproduce on this runtime.** It originated in a Flask
service wrapping NeMo PyTorch (`CacheAwareStreamingAudioBuffer` + `perform_streaming`), where
appending ~0.5 s of SoX silence improved recognition of final words. On NeMo-Speech.cpp the same
model produces complete transcripts with no padding. This runtime's `finalize()` already flushes
the tail with `chunk_size_mel + shift_size_mel` padded mel frames — **73 frames ≈ 730 ms at R=3**,
more than the manual 0.5 s workaround supplied.

Evidence strength: 2 short clips judged by reading, no references. Encouraging, not conclusive.

**Session silence is confirmed correct.** The gaps between finals (5.0 s, 4.4 s, 1.6 s) and the
13.5 s after the last final were deliberate pauses, not missed speech. No audio was dropped and the
ring buffer never overflowed; the endpointer fired on every pause as designed.

**RNN-T word timestamps are emission markers, not acoustic spans.** In
`RnntGreedyDecoder::present_token`, `cur_.start_frame = frame` is set when a word begins
(`src/asr/decoders/rnnt_greedy_decoder.cpp:193`) and `cur_.end_frame = frame + 1` is **re-assigned on
every non-punctuation subword token** of that word (`:199`). So a word's span covers the encoder
frames its subword pieces were emitted across, floored at one frame = **80 ms**
(`ms_per_enc_frame`, `src/asr/model.h:65-68`; converted at `src/asr/recognizer.cpp:450-451`).
Punctuation attaches without extending the span (`:197-198`).

Measured against the HTTP response for a 3.58 s clip, which matches this exactly:

| Word | Span | Frames |
|---|---|---|
| और | 1.60 – 1.68 | 1 (single token → nominal 80 ms) |
| दिक्कत | 1.84 – 2.24 | 5 (multi-subword) |
| है | 2.88 – 2.96 | 1 |
| नहीं | 3.04 – 3.12 | 1 |
| खाई | 3.52 – 3.68 | 2 |

The consequence stands: these are *emission* frames, not acoustic onsets/offsets. Single-token words
collapse to a nominal 80 ms however long they were actually spoken, and greedy RNN-T emits late and
in bursts — so the live session's `0:17.3 - 0:17.9` for six words was this effect, not a defect.

Consequences to respect rather than fix:

- **Subtitle output (SRT/VTT) from this model will have compressed, late cues.**
- **Any latency metric derived from word times is invalid** — use wall-clock arrival of results
  instead.
- **Diarization word-tagging consumes these timings** (`speaker_for_word_time`,
  `src/asr/recognizer.cpp:491`), so speaker attribution inherits the skew.

The TDT head does this properly — `cur_.end_frame = f + std::max(1, skip)` uses the predicted
duration (`rnnt_greedy_decoder.cpp:468`). Getting true word extents would mean a TDT checkpoint, not
a code change. Transcript *text* is unaffected.

**Latent inconsistency recorded, deliberately not fixed.** `CacheStreamRunner::finalize()`
(`src/asr/runner.cpp:1172`) pads with `0.0f` into a buffer holding raw un-normalized log-mel, where
true digital silence is `log(2^-24) ≈ −16.6`. So the pad is acoustically a flat moderately-loud
spectrum rather than silence, and those frames are fully attended (the attention mask covers only
the left cache warm-up region, `:857-868`). The repo's own `VadMaskerCfg::mask_value` already uses
`-16.635` for exactly this purpose. **No change made**: there is no evidence of user-visible harm at
R=3, and changing it would alter model behavior without measurement.

## Server (Phase 4) — already available in the prebuilt binary

Building from source was unnecessary and in fact blocked: `scripts/configure.sh cuda-server` failed
on this machine with `CMake was unable to find a build program corresponding to "Ninja"` plus unset
C/CXX compilers. The installed `nemo-speech` is the installer's **prebuilt** binary and already
ships with `NEMO_SPEECH_BUILD_HTTP` enabled, so `serve` works as-is:

```bash
nemo-speech serve --asr-model hindi-q8.gguf --asr.streaming.rnnt_right_context 3
```

```
[asr] model=hindi-q8 head=rnnt backend=CUDA0
HTTP API listening on http://127.0.0.1:8080
Playground: http://127.0.0.1:8080/
```

Note: a CUDA *source* build would additionally need the full CUDA toolkit (`nvcc`), not just the
runtime that `backend=CUDA0` uses. Not required while the prebuilt binary suffices.

### Critical: the realtime path has no server-side default ASR language

`serve` has **no `--language` flag for ASR**. The `default_language_code` at `app/serve.cpp:607` is
TTS-only. The realtime WebSocket takes the language from the client's `session.update` event
(`options.language_code = updated->string_or("language", ...)`, `server/http/http_server.cpp:1121+`),
defaulting to empty — which `prompt_index_for_lang` resolves to **`"auto"`**, not the fine-tuned
`hi-IN` prompt slot.

**Every realtime client must send `language: "hi-IN"` in `session.update` before any audio.** Config
is frozen once audio starts ("session configuration cannot change after audio starts"). The
file-upload endpoint takes it as a `language` form field instead (`http_server.cpp:649,656`).

The built-in playground does send it, from its `asr-lang` field — so that field must be set to
`hi-IN` in the UI or the session silently uses the `auto` prompt.

### Realtime protocol (already matches the Phase 7 requirement)

OpenAI-realtime shaped, at `/v1/audio/transcriptions/realtime`:

| Required event | Actual event |
|---|---|
| `session_started` | `session.created` / `session.updated` |
| `partial_transcript` | `conversation.item.input_audio_transcription.delta` |
| `final_transcript` | `conversation.item.input_audio_transcription.completed` |
| `endpoint_detected` | `input_audio_buffer.committed` |
| `error` | `error` |

Client sends `session.update`, then raw **Int16 PCM binary frames** (or base64 via
`input_audio_buffer.append`), and `input_audio_buffer.commit` to force an endpoint.

**Deltas are incremental, not snapshots** — the playground appends them
(`live.partial += m.delta`), and the server does prefix-delta diffing (`http_server.cpp:1053-1060`).
This already satisfies "clients can distinguish revisions of a partial from committed final text"
without concatenating partials as new sentences. No protocol work needed.

### `/v1/audio/transcriptions` is NOT the streaming path (important for evaluation)

The file-upload endpoint calls `Recognizer::recognize()` (`server/http/http_server.cpp:693`), which
**chooses the runner by audio length** (`src/asr/recognizer.cpp:630-647`):

```cpp
const bool use_streaming =
    (exceeds_offline_limit || (vulkan && head_kind != Ctc)) && supports_streaming;
```

- Within the positional-encoding budget → **`OfflineRunner`**: one non-cache-aware encoder pass over
  the whole file. Not streaming.
- Over the budget → `CacheStreamRunner`, i.e. internally chunked, and past the limit `OfflineRunner`
  would instead split at quiet points.
- Vulkan + non-CTC also forces the cache-aware graph. **CUDA does not**, so on this deployment
  normal-length files take the offline path.

The budget is `encoder_frames > enc.pos_emb_max_len` (`src/asr/recognizer.cpp:87-95`); at
subsampling 8 that is one frame per 80 ms of audio.

Either way the HTTP response is a **single** result — the client never receives incremental events,
so from the caller's perspective this endpoint is always non-streaming.

**Crucially, "offline" here does not mean full-context attention.** For a cache-aware-trained model
the converter sets `offline_left_ctx, offline_right_ctx = train_left_ctx, train_right_ctx`
(`conversion/asr.py:459-460`), so this model carries **offline [56, 3]** as well. The offline
encoder then builds a chunked-limited mask from those values
(`src/asr/encoder/fastconformer.cpp:1319-1353`), windowing attention over *chunks* rather than a
per-frame band: `left_chunks = L / (R + 1)`. The model never sees unbounded future context on
either path.

So offline-vs-streaming for this model is **not** a difference in attention budget. The real
differences are cache warm-up (streaming begins with empty K/V caches, masked via `attn_mask`),
tail handling (streaming depends on the `finalize()` flush; offline sees real audio to the end), and
chunk-boundary effects. That makes the two paths much closer than "offline vs streaming" suggests —
consistent with the identical transcripts observed on the test clip.

Which path ran is logged as (`src/asr/recognizer.cpp`, `log_execution_status`):

```
[asr] mode=offline head=rnnt attention-left=56 attention-right=3 encoder-frames
[asr] mode=streaming head=rnnt left=56 center=1 right=3 attention=60 encoder-frames step=320ms
```

`attention=full-context` appears only when both offline context values are negative — not the case
for this model. **Note the line is wrapped in `std::call_once` per path**, so it prints only on the
*first* offline request and the *first* streaming request of the process. A later request logs
nothing; restart the server to see it again.

Two consequences:

1. **`--asr.streaming.rnnt_right_context 3` has no effect on this endpoint** for normal-length
   audio. It configures cache-aware streaming geometry, which the offline runner does not use. It
   matters for the realtime WebSocket (and the long-audio fallback).
2. **File-endpoint WER is not streaming WER.** Comparing `curl` results against realtime results is
   comparing two different encoder paths. Use `transcribe --stream` or the realtime WebSocket for
   streaming numbers.

`Recognizer::recognize` calls `log_execution_status(use_streaming)` (`:648`), so the server's stderr
reports which path each request took — that is the authoritative check, not arithmetic.

### Deployment findings for the voice-bot use case

**`serve` handles SIGINT and SIGTERM correctly** (`app/serve.cpp:631-636`): both are trapped, a
watcher thread calls `http_server->stop()`, and the previous handlers are restored via RAII. No
change needed. (The missing-SIGTERM gap is in `transcribe --live` only — see gap 1 below — and does
not apply to a server deployment.)

**In-flight utterances are discarded on disconnect or shutdown.** The realtime read loop `break`s
and the handler lambda returns (`server/http/http_server.cpp:1197-1198`) without calling
`stream->finish()`; the stream is simply destroyed. For a client disconnect this is unavoidable —
there is no peer left to receive a final. On graceful shutdown the socket is being torn down anyway.
Recorded as accepted behavior, not a defect. If abandoned utterances need to be captured for
analytics, that is an application-level concern, not a runtime one.

**Concurrency ceiling is a hard throw, not a queue — the real production risk.** Each streaming
session owns one cache-aware encoder state slot, and the arena is sized by
`BatchingConfig::state_arena_slots`, **default 16** (`src/asr/batching.h:41`). Slot exhaustion
raises `"CacheAwareEncoder: state arena is full"`
(`src/asr/encoder/cache_aware_encoder.cpp:307-308`) rather than queueing or shedding politely, so
the 17th concurrent realtime session fails. Configurable via
`--asr.batching.state_arena_slots N` (or the YAML equivalent), and exposed on the C ABI as
`nemo_speech_asr_batching_config.state_arena_slots`. **Set this deliberately for the expected
concurrent-caller count before deploying**, and decide how the client should present a rejected
session.

**No ASR language default on the server** — see the realtime section above. Every client must send
`language: "hi-IN"` in `session.update`.

## Known gaps (not yet addressed)

1. **Signal handling** — `SignalHandlerGuard` is installed *after* model load and warmup
   (`app/transcribe.cpp:366`), so Ctrl+C during load kills the process outright. **`SIGTERM` is
   never handled** in live mode, so a container stop skips the entire tail-flush/`finish()` path.
   (`serve` handles both, `app/serve.cpp:631-635`.)
2. **No mic device selection** — hardcoded to the OS default (`ma_device_init(nullptr, ...)`).
   `examples/transcribe_live.cpp:68-78,219` already demonstrates enumeration plus `--device N`.
3. **`force_endpoint()` unreachable from the live CLI** — no keypress binding, though both servers
   expose it.
4. **Endpointing policy fields absent** from `VadEndpointerCfg`: min utterance duration, max
   utterance duration, min speech duration before an endpoint is accepted. (Pre-roll is unnecessary
   — audio is continuous and encoder context survives EOU. Post-speech hangover *is*
   `stop_history_eou_ms`.)
5. **Silent capture overflow** — no counter or log when the 2 s ring drops samples.
6. **Finalization pad value/duration** not configurable.

## Unresolved questions

1. Is there a Hindi evaluation set with reference transcripts? Blocking for any WER/CER number.
   Ideally it includes the clips that previously dropped final words in the PyTorch service.
2. Keep `q8_0`, or also build `f16`/`bf16` to separate quantization error from streaming error?
3. `asr.preprocessor.dither` is written by the converter but never read by any C++ code. Confirm
   what NeMo applies at eval time before treating this as a divergence.

## Next milestone

Run a live mic session end to end and judge it against the actual requirement:

```bash
nemo-speech transcribe --live --model hindi-q8.gguf \
  --asr.streaming.rnnt_right_context 3 -l hi-IN --stop-history-eou-ms 800
```

Specifically: do partials appear while speaking, does a final land after each pause, does the
session survive multiple utterances without restart. That determines whether the remaining work is
the gap list above or something larger.
