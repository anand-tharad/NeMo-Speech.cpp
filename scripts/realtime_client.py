#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal realtime ASR client for nemo-speech serve.

Streams either a WAV file or a live microphone over the realtime WebSocket and
prints partial and final transcripts separately, so partial revisions stay
visibly distinct from committed finals.

Protocol (server/http/http_server.cpp:1100-1185):
  -> session.update            must be first; config freezes once audio starts
  -> binary frames             raw PCM16 at the declared sample_rate
  -> input_audio_buffer.commit forces finish() and RESETS the stream
  <- session.created / session.updated
  <- conversation.item.input_audio_transcription.delta      (incremental text)
  <- conversation.item.input_audio_transcription.completed  (final)
  <- input_audio_buffer.committed / input_audio_buffer.cleared / error

Automatic endpointing emits .completed while preserving encoder state. An
explicit commit does not: it calls finish() and resets the stream, discarding
acoustic context. Use commit at end of input, not as an utterance separator.

Requires: pip install websockets        (file mode)
          pip install sounddevice       (mic mode; needs libportaudio)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
import wave

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

# The language tag must match the prompt the model was fine-tuned with. The
# server has no default: omitting it resolves to the "auto" prompt slot, which
# is a different set of weights than hi-IN.
DEFAULT_LANGUAGE = "hi-IN"
MIC_RATE = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("wav", nargs="?", help="16-bit PCM mono WAV file")
    src.add_argument("--mic", action="store_true", help="capture the microphone until Ctrl-C")
    src.add_argument("--list-devices", action="store_true", help="list input devices and exit")

    p.add_argument(
        "--url",
        default="ws://127.0.0.1:8082/v1/audio/transcriptions/realtime",
        help="RC6 realtime WebSocket. NOTE: this is the nemo-speech server, "
             "not the orchestrator on 8080 -- the orchestrator takes complete "
             "uploads and has no WebSocket, so this path has no retry logic.")
    p.add_argument("--language", default=DEFAULT_LANGUAGE, help="prompt tag (default: hi-IN)")
    p.add_argument("--device", type=int, help="mic input device index (see --list-devices)")
    p.add_argument(
        "--chunk-ms", type=int, default=160, help="audio frame size sent per message (default: 160)"
    )
    p.add_argument(
        "--endpointing-ms",
        type=float,
        default=800.0,
        help="trailing-silence EOU threshold for this session (default: 800)",
    )
    p.add_argument("--word-timestamps", action="store_true", help="request word timings")
    p.add_argument(
        "--json",
        action="store_true",
        help="emit NDJSON events on stdout (one object per line) and keep status on stderr, "
        "so the stream can be piped into another process",
    )
    p.add_argument(
        "--queue-frames",
        type=int,
        default=64,
        help="mic backpressure: bounded queue depth in frames (default: 64 ~ 10s)",
    )
    p.add_argument(
        "--no-pacing", action="store_true", help="file mode: send as fast as possible"
    )
    p.add_argument("--api-key", help="appended as ?api_key= when the server requires one")
    return p.parse_args()


def list_devices() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("pip install sounddevice")
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{index}] {dev['name']}  ({dev['max_input_channels']} ch)")
    return 0


def read_pcm16_mono(path: str) -> tuple[bytes, int]:
    """Return (raw PCM16 bytes, sample_rate). Rejects anything needing conversion."""
    with contextlib.closing(wave.open(path, "rb")) as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        if w.getnchannels() != 1:
            raise SystemExit(
                f"{path}: need mono, got {w.getnchannels()} channels — convert with:\n"
                f"  ffmpeg -i {path} -ac 1 -ar 16000 -c:a pcm_s16le mono.wav"
            )
        rate = w.getframerate()
        if not 8000 <= rate <= 96000:
            raise SystemExit(f"{path}: sample rate {rate} outside the server's 8000-96000 range")
        return w.readframes(w.getnframes()), rate


def normalize_ws_url(url: str) -> str:
    """Map http(s):// to ws(s):// so tunnel URLs can be pasted verbatim."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    if url.startswith(("ws://", "wss://")):
        return url
    raise SystemExit(f"--url must start with ws://, wss://, http:// or https:// — got {url!r}")


def emit(state: dict, **fields) -> None:
    """Write one NDJSON event to stdout and flush, so a reading process sees it
    immediately rather than when the pipe buffer fills."""
    fields["t"] = round(time.monotonic() - state["t0"], 3)
    print(json.dumps(fields, ensure_ascii=False), flush=True)


async def receive(ws, state: dict) -> None:
    """Drain server events until the stream is committed or errors out."""
    as_json = state["json"]
    async for raw in ws:
        event = json.loads(raw)
        kind = event.get("type", "")

        if kind.endswith(".delta"):
            # Deltas are incremental, not snapshots — accumulate them. The
            # server already does prefix-delta diffing, so appending is correct.
            state["partial"] += event.get("delta", "")
            if as_json:
                emit(state, type="partial", text=state["partial"])
            else:
                print(f"\r  partial: {state['partial']}", end="", flush=True)

        elif kind.endswith(".completed"):
            text = event.get("transcript", "")
            words = event.get("words") or []
            state["finals"].append(text)
            state["partial"] = ""
            if as_json:
                emit(state, type="final", text=text, words=words)
            else:
                elapsed = time.monotonic() - state["t0"]
                # Clear the partial line before printing the committed final.
                print(f"\r{' ' * 100}\r  FINAL [{elapsed:6.2f}s]: {text}", flush=True)
                for w in words:
                    print(f"           {w['start']:6.2f}-{w['end']:6.2f}  {w['word']}")

        elif kind == "session.created":
            s = event.get("session", {})
            if as_json:
                emit(state, type="session", model=s.get("model"), sample_rate=s.get("sample_rate"))
            else:
                print(f"  session.created  model={s.get('model')} rate={s.get('sample_rate')}")

        elif kind == "session.updated":
            if not as_json:
                print(f"  session.updated  language={state['language']}")

        elif kind == "input_audio_buffer.committed":
            if as_json:
                emit(state, type="committed")
            else:
                print("  committed (stream finished and reset)")
            return

        elif kind == "error":
            message = event.get("error", {}).get("message", raw)
            state["failed"] = True
            if as_json:
                emit(state, type="error", message=message)
            else:
                print(f"\n  ERROR: {message}", file=sys.stderr)
            return


async def send_session_update(ws, args: argparse.Namespace, rate: int) -> None:
    """Must be the first message: configuration freezes once audio starts."""
    await ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "sample_rate": rate,
                    "language": args.language,
                    "automatic_punctuation": True,
                    "word_timestamps": args.word_timestamps,
                    "endpointing_ms": args.endpointing_ms,
                },
            }
        )
    )


async def stream_file(ws, args: argparse.Namespace, state: dict, reader: asyncio.Task) -> None:
    pcm, rate = state["pcm"], state["rate"]
    frame_bytes = max(2, (rate * args.chunk_ms // 1000) * 2)
    for offset in range(0, len(pcm), frame_bytes):
        if reader.done():
            return
        await ws.send(pcm[offset : offset + frame_bytes])
        if not args.no_pacing:
            # Pace at real time so partials arrive like a live microphone.
            await asyncio.sleep(args.chunk_ms / 1000.0)


async def stream_mic(ws, args: argparse.Namespace, state: dict, reader: asyncio.Task) -> None:
    """Capture the mic into a bounded queue and forward frames to the server.

    Backpressure: the queue is bounded. The audio callback never blocks and never
    allocates unboundedly — if the socket falls behind, the OLDEST frame is
    dropped and counted, keeping latency bounded rather than growing a backlog.
    This mirrors the CLI's fixed ring buffer, except drops are reported.
    """
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("pip install sounddevice")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=args.queue_frames)
    blocksize = MIC_RATE * args.chunk_ms // 1000

    def offer(data: bytes) -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
                state["dropped"] += 1
        queue.put_nowait(data)

    def callback(indata, _frames, _time, status) -> None:
        if status:
            print(f"\n  [audio] {status}", file=sys.stderr)
        loop.call_soon_threadsafe(offer, bytes(indata))

    stream = sd.RawInputStream(
        samplerate=MIC_RATE,
        channels=1,
        dtype="int16",
        blocksize=blocksize,
        device=args.device,
        callback=callback,
    )
    name = sd.query_devices(args.device if args.device is not None else sd.default.device[0])["name"]
    print(
        f"  LIVE  {name} @ {MIC_RATE} Hz — speak, then Ctrl-C to finish\n",
        file=sys.stderr if state["json"] else sys.stdout,
    )

    with stream:
        while not reader.done() and not state["stop"]:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            await ws.send(frame)


async def main() -> int:
    args = parse_args()
    if args.list_devices:
        return list_devices()

    rate = MIC_RATE
    state: dict = {
        "partial": "",
        "finals": [],
        "dropped": 0,
        "stop": False,
        "failed": False,
        "json": args.json,
    }
    # In JSON mode stdout carries only NDJSON events, so status goes to stderr.
    status = sys.stderr if args.json else sys.stdout

    if not args.mic:
        state["pcm"], rate = read_pcm16_mono(args.wav)
        state["rate"] = rate
        print(f"{args.wav}: {len(state['pcm']) / 2 / rate:.2f}s @ {rate} Hz", file=status)

    # Accept http(s):// as a convenience — tunnels (ngrok, Cloudflare) advertise
    # an https:// URL, but a WebSocket must be dialled as wss://. Same port and
    # path, only the scheme differs.
    url = normalize_ws_url(args.url)
    if args.api_key:
        url += ("&" if "?" in url else "?") + f"api_key={args.api_key}"
    print(f"connecting {url}", file=status)

    # Tunnels on free plans may serve an HTML interstitial to anything that looks
    # like a browser; this header opts out on ngrok and is harmless elsewhere.
    # The keyword for it was renamed between websockets releases (extra_headers
    # before 13, additional_headers after), and older builds accept neither on
    # this path -- so probe rather than pin a version.
    headers = {"ngrok-skip-browser-warning": "true"}
    connect_kwargs = {"max_size": None}
    import inspect
    try:
        params = inspect.signature(websockets.connect).parameters
        if "additional_headers" in params:
            connect_kwargs["additional_headers"] = headers
        elif "extra_headers" in params:
            connect_kwargs["extra_headers"] = headers
    except (TypeError, ValueError):
        pass  # signature unavailable; connect without the header

    async with websockets.connect(url, **connect_kwargs) as ws:
        state["language"] = args.language
        state["t0"] = time.monotonic()
        reader = asyncio.create_task(receive(ws, state))

        await send_session_update(ws, args, rate)

        try:
            if args.mic:
                await stream_mic(ws, args, state, reader)
            else:
                await stream_file(ws, args, state, reader)
        except KeyboardInterrupt:
            pass

        # Flush the tail and get the end-of-stream final. This calls finish()
        # and resets the stream server-side; correct at end of input.
        if not reader.done():
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(reader, timeout=30)

        if not args.json:
            print("\n--- transcript ---")
            print(" ".join(t for t in state["finals"] if t))
        if state["dropped"]:
            print(
                f"\nWARNING: dropped {state['dropped']} mic frames "
                f"({state['dropped'] * args.chunk_ms / 1000:.1f}s) — the socket or server "
                f"could not keep up with capture.",
                file=sys.stderr,
            )
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
