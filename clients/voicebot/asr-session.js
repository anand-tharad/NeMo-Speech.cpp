// SPDX-License-Identifier: Apache-2.0
//
// Browser ASR session for a voice bot, against nemo-speech serve's realtime
// WebSocket (/v1/audio/transcriptions/realtime).
//
// Protocol notes that drive the design (server/http/http_server.cpp:1100-1185):
//
//   * session.update MUST be the first message. Configuration freezes as soon
//     as any audio arrives ("session configuration cannot change after audio
//     starts"), so language and endpointing are set once, up front.
//
//   * `language` has NO server-side default. Omitting it resolves to the
//     "auto" prompt slot, which for a fine-tuned model is a DIFFERENT set of
//     weights than the tag it was tuned with. Always send it.
//
//   * There is no speech_started/speech_stopped on this endpoint (those are
//     S2S/VoiceChat only). Turn detection comes from the events themselves:
//     first `.delta` = user started talking, `.completed` = user finished.
//     `.completed` is fired by server-side endpointing after
//     `endpointing_ms` of trailing silence. That is the bot's cue to reply.
//
//   * input_audio_buffer.commit calls finish() AND resets the stream, which
//     discards encoder cache. Do NOT use it as a per-turn separator — let
//     endpointing emit `.completed` and keep the acoustic context. Use
//     `clear` to discard audio instead (see mute()).
//
//   * `.delta` payloads are incremental text, not snapshots. Accumulate them.
//
// One WebSocket is held for the whole conversation. Opening a socket per turn
// costs a handshake and throws away encoder context mid-dialogue.

const WORKLET = `
class Pcm16 extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.target = options.processorOptions.frameSamples;
    this.buf = new Int16Array(this.target);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.target) {
        // Transfer a copy; the worklet keeps reusing its own buffer.
        const out = this.buf.slice();
        this.port.postMessage(out, [out.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm16', Pcm16);
`;

const MODEL_RATE = 16000;

export class AsrSession {
  /**
   * @param {object}   opts
   * @param {string}   opts.url            ws:// or wss:// realtime endpoint
   * @param {string}   opts.language       prompt tag, e.g. "hi-IN" (required)
   * @param {number}  [opts.endpointingMs] trailing-silence turn threshold
   * @param {number}  [opts.frameMs]       audio frame size sent per message
   * @param {string}  [opts.apiKey]        when the server requires one
   * @param {function}[opts.onPartial]     (text) => void, revisable
   * @param {function}[opts.onFinal]       (text, words) => void, committed turn
   * @param {function}[opts.onState]       ("connecting"|"listening"|"closed") => void
   * @param {function}[opts.onError]       (Error) => void
   */
  constructor(opts) {
    if (!opts.language) throw new Error('language is required: the server has no default');
    this.opts = { endpointingMs: 800, frameMs: 160, ...opts };
    this.socket = null;
    this.context = null;
    this.media = null;
    this.node = null;
    this.partial = '';
    this.muted = false;
    this.closing = false;
  }

  get listening() {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  async start() {
    if (this.socket) return;
    this.closing = false;
    this.#state('connecting');

    this.media = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      video: false,
    });

    // Ask the browser for the model's rate so it resamples properly in native
    // code. Not every browser honours the hint, so read back the rate actually
    // granted and declare THAT to the server, which resamples 8k-96k itself.
    this.context = new AudioContext({ sampleRate: MODEL_RATE });
    const rate = this.context.sampleRate;

    const url = this.opts.url + (this.opts.apiKey
      ? (this.opts.url.includes('?') ? '&' : '?') + 'api_key=' + encodeURIComponent(this.opts.apiKey)
      : '');
    this.socket = new WebSocket(url);
    this.socket.binaryType = 'arraybuffer';

    this.socket.onmessage = (e) => this.#onEvent(JSON.parse(e.data));
    this.socket.onerror = () => this.#fail(new Error('websocket error'));
    this.socket.onclose = () => { this.#teardownAudio(); this.#state('closed'); };

    await new Promise((resolve, reject) => {
      this.socket.onopen = resolve;
      this.socket.addEventListener('close', () => reject(new Error('closed before open')), { once: true });
    });

    // First message, before any audio.
    this.#send({
      type: 'session.update',
      session: {
        sample_rate: rate,
        language: this.opts.language,
        automatic_punctuation: true,
        word_timestamps: true,
        endpointing_ms: this.opts.endpointingMs,
      },
    });

    await this.#startCapture(rate);
    this.#state('listening');
  }

  async #startCapture(rate) {
    const blob = new Blob([WORKLET], { type: 'application/javascript' });
    const moduleUrl = URL.createObjectURL(blob);
    try {
      await this.context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }

    const frameSamples = Math.max(128, Math.round((rate * this.opts.frameMs) / 1000));
    this.node = new AudioWorkletNode(this.context, 'pcm16', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { frameSamples },
    });
    this.node.port.onmessage = (e) => {
      // Dropping while muted is deliberate: see mute().
      if (this.muted || this.socket?.readyState !== WebSocket.OPEN) return;
      this.socket.send(e.data.buffer);
    };
    this.context.createMediaStreamSource(this.media).connect(this.node);
  }

  #onEvent(event) {
    const type = event.type || '';
    if (type.endsWith('.delta')) {
      // Incremental, not a snapshot.
      this.partial += event.delta || '';
      this.opts.onPartial?.(this.partial);
    } else if (type.endsWith('.completed')) {
      // Server-side endpointing closed a turn. This is the bot's cue.
      const text = event.transcript || '';
      this.partial = '';
      this.opts.onPartial?.('');
      if (text.trim()) this.opts.onFinal?.(text, event.words || []);
    } else if (type === 'error') {
      this.#fail(new Error(event.error?.message || 'server error'));
    }
    // session.created / session.updated / input_audio_buffer.* need no action.
  }

  /**
   * Stop sending audio without tearing down the session. Use this while the bot
   * is speaking so its own TTS is not transcribed back as user speech, then
   * unmute() when it finishes. Preferred over commit(), which would reset the
   * server's encoder state. `clear` discards whatever the server had buffered.
   */
  mute() {
    if (this.muted) return;
    this.muted = true;
    this.partial = '';
    this.opts.onPartial?.('');
    this.#send({ type: 'input_audio_buffer.clear' });
  }

  unmute() {
    this.muted = false;
  }

  /**
   * Force a turn boundary now instead of waiting for endpointing — e.g. the user
   * taps "send". NOTE: this resets the server stream and discards encoder
   * context, so prefer letting endpointing fire during continuous dialogue.
   */
  commit() {
    this.#send({ type: 'input_audio_buffer.commit' });
  }

  /** End the conversation and release the microphone. */
  stop() {
    this.closing = true;
    if (this.socket?.readyState === WebSocket.OPEN) {
      // Flush the tail so the last utterance is not lost, then close.
      this.#send({ type: 'input_audio_buffer.commit' });
      this.socket.close();
    }
    this.#teardownAudio();
  }

  #send(message) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  #fail(error) {
    if (!this.closing) this.opts.onError?.(error);
  }

  #teardownAudio() {
    this.node?.port.close();
    this.node?.disconnect();
    this.media?.getTracks().forEach((t) => t.stop());
    this.context?.close().catch(() => {});
    this.node = this.media = this.context = null;
    this.socket = null;
  }

  #state(name) {
    this.opts.onState?.(name);
  }
}
