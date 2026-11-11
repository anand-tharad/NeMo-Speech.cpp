// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Decoder unit tests - model-free, synthetic inputs. Covers the contracts the
// runners and endpointing depend on:
//
//   GreedyCtcDecoder   collapse rule across chunk boundaries, blank handling,
//                      last_emit_frame exactness, utterance confidence, word
//                      timings, reset_utterance vs reset semantics.
//   RnntGreedyDecoder  scripted MockRnntEngine: emission sequence, blank =
//                      frame advance, max_symbols_per_step cap, LSTM state /
//                      prev_token preserved across reset_utterance (riva EOU).
//
// Usage: ./test_decoders   (no model files, no GPU)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "greedy_ctc_decoder.h"
#include "rnnt_greedy_decoder.h"

using namespace nemo_speech::asr;

static int g_fail = 0;
static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

static bool
near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// GreedyCtcDecoder. 4 classes: 0="▁hel", 1="lo", 2="▁world", 3=<blank>.
// Frames are log-prob rows; emit_frame() builds a row whose argmax is `id`
// with softmax probability `p` (the rest share 1-p uniformly).
// ---------------------------------------------------------------------------
namespace {

constexpr int kClasses = 4;
constexpr int kBlank = 3;

std::vector<float>
frame_for(int id, float p = 0.8f) {
    std::vector<float> row(kClasses);
    const float rest = (1.0f - p) / (kClasses - 1);
    for (int c = 0; c < kClasses; c++) row[c] = std::log(c == id ? p : rest);
    return row;
}

std::vector<float>
frames_for(const std::vector<int>& ids, float p = 0.8f) {
    std::vector<float> out;
    for (int id : ids) {
        auto row = frame_for(id, p);
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

GreedyCtcDecoder
make_greedy() {
    CtcConfig cfg;
    cfg.d_model = 8;  // unused by the decoder
    cfg.num_classes = kClasses - 1;
    cfg.blank_id = kBlank;
    return GreedyCtcDecoder(cfg, {"\xE2\x96\x81hel", "lo", "\xE2\x96\x81world"});
}

void
test_sentencepiece_punctuation_detokenization() {
    const std::vector<std::string> vocab = {
        "\xE2\x96\x81"
        "How",
        "\xE2\x96\x81old",
        "\xE2\x96\x81is",
        "\xE2\x96\x81"
        "Brooklyn",
        "\xE2\x96\x81"
        "Bridge",
        "\xE2\x96\x81?",
        "\xE2\x96\x81!",
        "\xE2\x96\x81\xE0\xA5\xA4"};
    check(
        detokenize_sentencepiece({0, 1, 2, 3, 4, 5}, vocab) == "How old is Brooklyn Bridge?",
        "sentencepiece: boundary-prefixed terminator attaches to preceding word");
    check(
        detokenize_sentencepiece({0, 6}, vocab) == "How!" &&
            detokenize_sentencepiece({0, 7}, vocab) == "How\xE0\xA5\xA4",
        "sentencepiece: ASCII and Devanagari terminators attach consistently");

    std::string incremental;
    append_sentencepiece_tokens(incremental, {0, 1}, vocab);
    append_sentencepiece_tokens(incremental, {2, 3, 4}, vocab);
    append_sentencepiece_tokens(incremental, {5}, vocab);
    check(
        incremental == detokenize_sentencepiece({0, 1, 2, 3, 4, 5}, vocab),
        "sentencepiece: incremental detokenization matches full detokenization");
}

void
test_greedy_collapse_and_blank() {
    auto dec = make_greedy();
    // [0, 0, blank, 0] -> repeat collapses, blank separates -> emits {0, 0}.
    auto fr = frames_for({0, 0, kBlank, 0});
    auto ids = dec.step(fr.data(), kClasses, 4, /*frame_offset=*/0);
    check(ids == std::vector<int>({0, 0}), "greedy: collapse rule + blank separation");
}

void
test_greedy_cross_chunk_collapse() {
    auto dec = make_greedy();
    auto a = frames_for({kBlank, 0});  // chunk 1 ends emitting id 0
    auto b = frames_for({0, kBlank});  // chunk 2 starts with the same id
    auto ids1 = dec.step(a.data(), kClasses, 2, 0);
    auto ids2 = dec.step(b.data(), kClasses, 2, 2);
    check(ids1 == std::vector<int>({0}), "greedy: chunk 1 emits once");
    check(ids2.empty(), "greedy: repeated id across chunk boundary collapses");
}

void
test_greedy_seam_dedup() {
    // Buffered-window seam: the same word straddling an emit boundary can
    // spike once per window with a blank between (each window encodes the
    // seam with slightly different context). A window-leading repeat of the
    // last committed token within kSeamMergeMaxGap(=2) frames is suppressed.
    {
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank});  // window 1: spike at frame 0
        auto b = frames_for({0, kBlank});  // window 2: re-spike at frame 2
        auto ids1 = dec.step(a.data(), kClasses, 2, 0);
        auto ids2 = dec.step(b.data(), kClasses, 2, 2);
        check(ids1 == std::vector<int>({0}), "greedy: seam window 1 emits once");
        check(ids2.empty(), "greedy: seam re-spike within gap suppressed");
        // Window 3 re-spikes again: the suppressed frame extended the
        // committed token's reach, so the chain keeps deduping.
        auto ids3 = dec.step(b.data(), kClasses, 2, 4);
        check(ids3.empty(), "greedy: chained seam re-spike suppressed");
    }
    {
        // A genuine blank-separated repeat beyond the seam gap still emits.
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank, kBlank, kBlank});
        auto b = frames_for({0, kBlank});
        dec.step(a.data(), kClasses, 4, 0);
        auto ids = dec.step(b.data(), kClasses, 2, 4);  // gap 4 > 2
        check(ids == std::vector<int>({0}), "greedy: distant repeat re-emits");
    }
    {
        // Mid-window blank-separated repeats are not seam artifacts and
        // still emit even within the gap.
        auto dec = make_greedy();
        auto fr = frames_for({0, kBlank, 0});
        auto ids = dec.step(fr.data(), kClasses, 3, 0);
        check(ids == std::vector<int>({0, 0}), "greedy: mid-window repeat unaffected");
    }
    {
        // reset_utterance clears the dedup state: the next utterance's first
        // token emits even as an immediate repeat.
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank});
        dec.step(a.data(), kClasses, 2, 0);
        dec.reset_utterance();
        auto ids = dec.step(a.data(), kClasses, 2, 2);
        check(ids == std::vector<int>({0}), "greedy: reset_utterance clears seam state");
    }
}

void
test_greedy_last_emit_frame() {
    auto dec = make_greedy();
    check(dec.last_emit_frame() == -1, "greedy: last_emit_frame -1 before audio");
    // Non-blank at local frames 1 and 2; offset 100 -> last speech = 102.
    // The repeat at frame 2 emits no token but is still speech evidence.
    auto fr = frames_for({kBlank, 0, 0, kBlank});
    dec.step(fr.data(), kClasses, 4, /*frame_offset=*/100);
    check(dec.last_emit_frame() == 102, "greedy: last_emit_frame = last non-blank frame");
}

void
test_greedy_confidence() {
    auto dec = make_greedy();
    // Two emissions at known softmax probs 0.8 and 0.6 -> mean 0.7.
    auto f1 = frame_for(0, 0.8f);
    auto f2 = frame_for(2, 0.6f);
    std::vector<float> fr;
    fr.insert(fr.end(), f1.begin(), f1.end());
    fr.insert(fr.end(), f2.begin(), f2.end());
    dec.step(fr.data(), kClasses, 2, 0);
    check(near(dec.confidence(), 0.7f, 1e-3f), "greedy: confidence = mean token softmax");
}

void
test_greedy_word_timings() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    // ▁hel(f0) lo(f1) blank ▁world(f3) -> words "hel"+"lo" [0,2) and "world" [3,4).
    auto fr = frames_for({0, 1, kBlank, 2});
    dec.step(fr.data(), kClasses, 4, 0);
    dec.finalize();  // flush the in-progress word
    const auto& w = dec.word_timings();
    bool ok = w.size() == 2 && w[0].word == "hello" && w[0].start_frame == 0 &&
              w[0].end_frame == 2 && w[1].word == "world" && w[1].start_frame == 3 &&
              w[1].end_frame == 4;
    check(ok, "greedy: word timings (boundary split, start/end frames)");
}

void
test_greedy_punctuation_word_timing() {
    CtcConfig cfg;
    cfg.d_model = 8;
    cfg.num_classes = kClasses - 1;
    cfg.blank_id = kBlank;
    GreedyCtcDecoder dec(cfg, {"\xE2\x96\x81hello", "\xE2\x96\x81?", "\xE2\x96\x81world"});
    dec.set_compute_timestamps(true);
    auto fr = frames_for({0, 1, kBlank, 2});
    dec.step(fr.data(), kClasses, 4, 0);
    dec.finalize();
    const auto& w = dec.word_timings();
    const bool ok = w.size() == 2 && w[0].word == "hello?" && w[0].start_frame == 0 &&
                    w[0].end_frame == 2 && w[1].word == "world";
    check(ok, "greedy: boundary-prefixed terminator stays with timed word");
}

void
test_greedy_reset_utterance_vs_reset() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    auto fr = frames_for({0});
    dec.step(fr.data(), kClasses, 1, 0);
    dec.finalize();
    dec.reset_utterance();
    check(dec.word_timings().empty(), "greedy: reset_utterance clears word timings");
    check(near(dec.confidence(), 1.0f), "greedy: reset_utterance clears confidence");
    check(dec.last_emit_frame() == 0, "greedy: reset_utterance keeps last_emit_frame");
    // reset_utterance clears the collapse state (utterance boundary): the same
    // id right after the EOU is a new utterance's first token and must emit.
    auto ids = dec.step(fr.data(), kClasses, 1, 1);
    check(ids == std::vector<int>({0}), "greedy: reset_utterance clears prev id (re-emits)");
    // Full reset also clears collapse state.
    dec.reset();
    check(dec.last_emit_frame() == -1, "greedy: reset clears last_emit_frame");
    ids = dec.step(fr.data(), kClasses, 1, 0);
    check(ids == std::vector<int>({0}), "greedy: reset clears prev id (re-emits)");
}

void
test_greedy_open_word_needs_finalize() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    // "▁hel" opens a word with no closing boundary piece - it stays open.
    auto fr = frames_for({0});
    dec.step(fr.data(), kClasses, 1, 0);
    check(dec.word_timings().empty(), "greedy: open trailing word absent before finalize");
    dec.finalize();
    const auto& w = dec.word_timings();
    check(
        w.size() == 1 && w[0].word == "hel",
        "greedy: finalize flushes the open trailing word into word_timings");
}

// ---------------------------------------------------------------------------
// RnntGreedyDecoder with a scripted staged engine. Each joint round supplies
// argmax ids for all remaining frames under one predictor state. Predictor h/c
// candidates are stamped with a per-call serial so blank caching and state
// threading are observable.
// ---------------------------------------------------------------------------
class MockRnntEngine : public RnntEngine {
   public:
    MockRnntEngine() {
        cfg_.d_model = 4;
        cfg_.vocab_size = 5;  // ids 0..3 + blank
        cfg_.blank_id = 4;
        cfg_.pred_hidden = 3;
        cfg_.pred_num_layers = 1;
        cfg_.joint_dim = 4;
        cfg_.max_symbols_per_step = 3;
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "\xE2\x96\x81"
            "c",
            "\xE2\x96\x81?"};
    }

    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }
    void set_piece(int id, const std::string& text) { vocab_.at(id) = text; }

    struct State : RnntStreamState {
        float bank[2] = {0.0f, 0.0f};
    };

    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        return std::make_unique<State>();
    }

    void predict_rnnt(RnntStreamState& opaque, int prev_token, int active_bank) override {
        auto& state = static_cast<State&>(opaque);
        last_prev_token = prev_token;
        last_h_in = state.bank[active_bank];
        predictor_calls++;
        serial += 1.0f;
        state.bank[active_bank ^ 1] = serial;
    }

    void joint_argmax(
        RnntStreamState& /*state*/, const float* /*enc_proj*/, int joint_dim, int T,
        int32_t* token_ids, const float* logit_bias = nullptr) override {
        check(joint_dim == cfg_.joint_dim, "rnnt mock: projected width");
        joint_calls++;
        if (logit_bias != nullptr) {
            last_logit_bias.assign(logit_bias, logit_bias + cfg_.vocab_size);
        } else {
            last_logit_bias.clear();
        }
        bias_log.push_back(last_logit_bias);
        std::vector<int> round;
        if (!rounds.empty()) {
            round = std::move(rounds.front());
            rounds.pop_front();
        }
        for (int i = 0; i < T; ++i) {
            token_ids[i] = (i < static_cast<int>(round.size())) ? round[i] : cfg_.blank_id;
        }
    }

    void joint_tdt_argmax(RnntStreamState&, const float*, int, int, int32_t*, int32_t*) override {
        throw std::runtime_error("RNNT mock does not implement TDT");
    }

    // Word boosting needs a tokenizer to build a non-empty tree. Map any phrase
    // onto a fixed two-token sequence so bias_on_ can be reached model-free.
    std::vector<int> encode_phrase(const std::string& text) const override {
        if (text.empty())
            return {};
        return {0, 1};
    }

    // Make the mock advertise itself as a Token-and-Duration Transducer, to
    // exercise the blank-penalty TDT guard.
    void set_durations(std::vector<int32_t> d) { cfg_.durations = std::move(d); }

    std::deque<std::vector<int>> rounds;
    int last_prev_token = -100;
    float last_h_in = -1.0f;
    float serial = 0.0f;
    int predictor_calls = 0;
    int joint_calls = 0;
    std::vector<float> last_logit_bias;
    // Every bias seen, in call order; an empty entry means nullptr was passed.
    // One step() can run two joint passes with different biases, so only a log
    // can show which pass received which.
    std::vector<std::vector<float>> bias_log;

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
};

void
test_rnnt_emission_and_blank() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    // T=3 frames. f0 emits 0 then blank; f1 blank; f2 emits 1, 2, blank.
    // Each vector is one speculative joint pass over the remaining frames.
    eng.rounds = {{0, blank, blank}, {blank, blank, 1}, {2}, {blank}};
    std::vector<float> enc(3 * 4, 0.0f);
    auto ids = dec.step(enc.data(), 4, 3, /*frame_offset=*/10);
    check(ids == std::vector<int>({0, 1, 2}), "rnnt: scripted emission sequence");
    check(dec.last_emit_frame() == 12, "rnnt: last_emit_frame = last emitting frame");
    check(eng.predictor_calls == 4, "rnnt: predictor runs only initially and after emits");
    check(eng.joint_calls == 4, "rnnt: joint consumes blank runs in vectorized rounds");
    const auto& stats = dec.stats();
    check(
        stats.encoder_frames == 3 && stats.emitted_tokens == 3 && stats.predictor_calls == 4 &&
            stats.joint_calls == 4 && stats.joint_frames == 8,
        "rnnt: staged execution counters");
}

void
test_rnnt_max_symbols_cap() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    // Script never returns blank: the inner loop must stop at
    // max_symbols_per_step (3) emissions for the single frame.
    eng.rounds = {{0}, {1}, {2}, {3}, {0}, {1}};
    std::vector<float> enc(4, 0.0f);
    auto ids = dec.step(enc.data(), 4, 1, 0);
    check(static_cast<int>(ids.size()) == 3, "rnnt: max_symbols_per_step caps the inner loop");
}

void
test_rnnt_finalization_preserves_native_punctuation() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);

    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.last_logit_bias.empty(), "rnnt: initial silence has no punctuation bias");

    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 1);
    eng.rounds = {{blank}, {3}};
    const auto tail = dec.step(enc.data(), 4, 1, 2);
    const int joints = eng.joint_calls;
    dec.finalize();
    check(
        tail.empty() && eng.last_logit_bias.empty() && eng.joint_calls == joints &&
            dec.stats().emitted_tokens == 1,
        "rnnt: a clipped stream closes without forcing terminal punctuation");

    dec.reset();
    dec.set_compute_timestamps(true);
    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    eng.rounds = {{3}, {blank}};
    const auto native = dec.step(enc.data(), 4, 1, 1);
    dec.finalize();
    check(
        native == std::vector<int>{3} && eng.last_logit_bias.empty() &&
            dec.word_timings().size() == 1 && dec.word_timings()[0].word == "a?",
        "rnnt: finalization preserves native terminal punctuation");
}

void
test_rnnt_stalls_do_not_inject_punctuation() {
    MockRnntEngine eng;
    eng.set_piece(2, ",");
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);
    for (bool formatting : {false, true}) {
        dec.reset();
        AsrRequestOptions opts;
        opts.enable_automatic_punctuation = formatting;
        opts.enable_word_time_offsets = true;
        dec.set_request_options(opts);
        eng.rounds = {{0}, {blank}};
        dec.step(enc.data(), 4, 1, 0);
        const int predictors = eng.predictor_calls;
        for (int frame = 1; frame <= 100; ++frame) {
            const int joints = eng.joint_calls;
            // A second query would return punctuation: it must never be made.
            eng.rounds = {{blank}, {3}};
            const auto tokens = dec.step(enc.data(), 4, 1, frame);
            check(
                tokens.empty() && eng.last_logit_bias.empty() && eng.joint_calls == joints + 1 &&
                    eng.predictor_calls == predictors && dec.stats().emitted_tokens == 1 &&
                    dec.last_emit_frame() == 0 && dec.last_speech_frame() == 0,
                "rnnt: prolonged stalls neither bias punctuation nor run a suggestion query");
        }
        eng.rounds = {{3}, {blank}};
        const auto native = dec.step(enc.data(), 4, 1, 101);
        dec.finalize();
        check(
            native == std::vector<int>{3} && eng.last_logit_bias.empty() &&
                dec.word_timings().size() == 1 && dec.word_timings()[0].word == "a?" &&
                dec.word_timings()[0].end_frame == 1,
            "rnnt: native punctuation survives prolonged stalls without changing word duration");
        dec.reset_utterance();
        for (int frame = 102; frame <= 110; ++frame) {
            eng.rounds = {{blank}, {3}};
            check(
                dec.step(enc.data(), 4, 1, frame).empty() && eng.last_logit_bias.empty(),
                "rnnt: soft EOU does not enable punctuation bias");
        }
        eng.rounds = {{blank}};
        dec.step(enc.data(), 4, 1, 111);
        dec.finalize();
        check(
            eng.last_logit_bias.empty() && dec.word_timings().empty(),
            "rnnt: closing after an endpoint cannot synthesize another punctuation final");
    }
}

void
test_rnnt_late_punctuation_returns_to_previous_final() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);
    auto step = [&](std::vector<int> round, int64_t frame) {
        eng.rounds = {std::move(round), {blank}};
        return dec.step(enc.data(), 4, 1, frame);
    };

    step({3}, 0);
    check(
        dec.take_late_punctuation().empty(),
        "rnnt: leading punctuation at stream start is dropped");

    step({0}, 1);
    dec.reset_utterance();
    check(step({3}, 2).empty(), "rnnt: late punctuation is not part of the next utterance");
    check(dec.take_late_punctuation().empty(), "rnnt: late punctuation waits for the next word");
    eng.set_piece(
        1,
        "\xE2\x96\x81"
        "B");
    step({1}, 3);
    check(
        dec.take_late_punctuation() == "?",
        "rnnt: late punctuation before a capitalized word returns to the previous final");
    check(dec.take_late_punctuation().empty(), "rnnt: late punctuation is delivered once");

    dec.reset_utterance();
    step({3}, 4);
    step({2}, 5);
    check(
        dec.take_late_punctuation().empty(),
        "rnnt: a late sentence mark before a lowercase word is dropped");

    step({3}, 6);
    step({0}, 7);
    check(dec.take_late_punctuation().empty(), "rnnt: punctuation inside an utterance is not late");

    dec.reset_utterance();
    step({3}, 8);
    check(
        dec.take_late_punctuation(/*end_of_stream=*/true) == "?",
        "rnnt: end of stream releases a waiting late mark");

    step({0}, 9);
    dec.reset_utterance();
    step({3}, 10);
    dec.reset();
    check(dec.take_late_punctuation().empty(), "rnnt: hard reset discards late punctuation");
}

void
test_rnnt_endpoint_punctuation_continuity() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    AsrRequestOptions opts;
    opts.enable_word_time_offsets = true;
    opts.enable_automatic_punctuation = true;
    dec.set_request_options(opts);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);
    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    for (int frame = 1; frame <= 8; ++frame) {
        eng.rounds = {{blank}};
        dec.step(enc.data(), 4, 1, frame);
    }
    check(eng.last_logit_bias.empty(), "rnnt: continuing silence never enables punctuation bias");
    eng.rounds = {{3}, {blank}};
    dec.step(enc.data(), 4, 1, 9);
    dec.finalize();
    check(
        dec.word_timings().size() == 1 && dec.word_timings()[0].word == "a?" &&
            dec.word_timings()[0].end_frame == 1 && dec.last_speech_frame() == 0,
        "rnnt: terminal punctuation stays in the published word");
    const int predictor_calls = eng.predictor_calls;
    dec.reset_utterance();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 10);
    check(
        eng.predictor_calls == predictor_calls && eng.last_logit_bias.empty(),
        "rnnt: soft EOU preserves cached predictor without injecting punctuation bias");
    eng.rounds = {{3}, {blank}};
    auto late = dec.step(enc.data(), 4, 1, 11);
    dec.finalize();
    check(
        late.empty() && dec.word_timings().empty() && eng.last_prev_token == 3,
        "rnnt: late punctuation updates predictor but creates no next-final text or word");
    check(
        dec.last_emit_frame() == 11 && dec.last_speech_frame() == 0,
        "rnnt: late punctuation cannot re-arm the endpoint speech clock");
    eng.rounds = {{1}, {blank}};
    auto next = dec.step(enc.data(), 4, 1, 12);
    eng.rounds = {{3}, {blank}};
    dec.step(enc.data(), 4, 1, 13);
    eng.rounds = {{3}, {blank}};
    auto repeated = dec.step(enc.data(), 4, 1, 14);
    dec.finalize();
    check(
        next == std::vector<int>{1} && repeated.empty() && dec.word_timings().size() == 1 &&
            dec.word_timings()[0].word == "b?" && dec.word_timings()[0].start_frame == 12 &&
            dec.word_timings()[0].end_frame == 13,
        "rnnt: next utterance retains text, punctuation and absolute timings without duplicates");
    dec.reset();
    check(dec.last_speech_frame() == -1, "rnnt: hard reset clears endpoint speech evidence");
    for (int frame = 0; frame < 8; ++frame) {
        eng.rounds = {{blank}};
        dec.step(enc.data(), 4, 1, frame);
    }
    check(eng.last_logit_bias.empty(), "rnnt: initial silence cannot accumulate punctuation bias");
    check(
        !sp_is_punctuation("-") && !sp_is_punctuation("'") && !sp_is_punctuation("\"") &&
            !sp_is_punctuation("don't") && !sp_is_punctuation("-5"),
        "rnnt: publication cleanup preserves signs, quotes and contractions");
    eng.set_piece(2, "\xE2\x96\x81");
    dec.reset();
    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    eng.rounds = {{2}, {blank}};
    dec.step(enc.data(), 4, 1, 1);
    eng.rounds = {{3}, {blank}};
    dec.step(enc.data(), 4, 1, 2);
    eng.rounds = {{1}, {blank}};
    dec.step(enc.data(), 4, 1, 3);
    dec.finalize();
    check(
        dec.word_timings().size() == 2 && dec.word_timings()[0].word == "a?" &&
            dec.word_timings()[1].word == "b",
        "rnnt: bare word separator cannot detach punctuation into its own timed word");
}

void
test_rnnt_state_threading_and_reset_utterance() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);

    // Frame emits token 2 (state serial 1 latched), then blank (serial 2
    // not latched: blank leaves the LSTM state untouched).
    eng.rounds = {{2}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.predictor_calls == 2, "rnnt: predictor candidate cached after blank");

    // Next call: the decoder must present the post-emit state (serial 1)
    // and prev_token=2.
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 1);
    check(eng.last_prev_token == 2, "rnnt: prev_token carried across steps");
    check(near(eng.last_h_in, 1.0f), "rnnt: LSTM state latched on emit only (not on blank)");
    check(eng.predictor_calls == 2, "rnnt: blank cache survives step boundary");

    // riva EOU semantics: reset_utterance keeps predictor state + prev_token.
    dec.reset_utterance();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 2);
    check(eng.last_prev_token == 2, "rnnt: prev_token survives reset_utterance");
    check(near(eng.last_h_in, 1.0f), "rnnt: LSTM state survives reset_utterance");
    check(dec.last_emit_frame() == 0, "rnnt: last_emit_frame survives reset_utterance");
    check(eng.predictor_calls == 2, "rnnt: blank cache survives reset_utterance");

    // Full reset: prev_token back to blank, state zeroed.
    dec.reset();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.last_prev_token == blank, "rnnt: reset restores prev_token = blank");
    check(near(eng.last_h_in, 0.0f), "rnnt: reset zeroes the LSTM state");
    check(dec.last_emit_frame() == -1, "rnnt: reset clears last_emit_frame");
}

void
test_rnnt_word_timings() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // f0 emits "▁a", f1 emits "▁b" -> two words at global frames 5 and 6.
    eng.rounds = {{0, blank}, {blank, 1}, {blank}};
    std::vector<float> enc(2 * 4, 0.0f);
    dec.step(enc.data(), 4, 2, /*frame_offset=*/5);
    dec.finalize();
    const auto& w = dec.word_timings();
    bool ok = w.size() == 2 && w[0].word == "a" && w[0].start_frame == 5 && w[1].word == "b" &&
              w[1].start_frame == 6;
    check(ok, "rnnt: word timings on scripted emissions");
}

void
test_rnnt_punctuation_word_timing() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // f0 emits "▁a", f1 emits "▁?": punctuation attaches without extending speech time.
    eng.rounds = {{0, blank}, {blank, 3}, {blank}};
    std::vector<float> enc(2 * 4, 0.0f);
    dec.step(enc.data(), 4, 2, /*frame_offset=*/5);
    dec.finalize();
    const auto& w = dec.word_timings();
    const bool ok =
        w.size() == 1 && w[0].word == "a?" && w[0].start_frame == 5 && w[0].end_frame == 6;
    check(ok, "rnnt: boundary-prefixed terminator stays with timed word");
}

void
test_rnnt_open_word_needs_finalize() {
    // The contract CacheStreamRunner::finalize() relies on: an open trailing
    // word reaches word_timings() only after finalize(). Without the runner
    // calling head_->finalize() at EOS, the last word's WordInfo is dropped.
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    eng.rounds = {{0}, {eng.rnnt_config().blank_id}};  // emit "▁a", then blank
    std::vector<float> enc(eng.rnnt_config().d_model, 0.0f);
    dec.step(enc.data(), eng.rnnt_config().d_model, 1, /*frame_offset=*/0);
    check(dec.word_timings().empty(), "rnnt: open trailing word absent before finalize");
    dec.finalize();
    const auto& w = dec.word_timings();
    check(
        w.size() == 1 && w[0].word == "a",
        "rnnt: finalize flushes the open trailing word into word_timings");
}

class MockTdtEngine : public RnntEngine {
   public:
    MockTdtEngine() {
        cfg_.d_model = cfg_.joint_dim = 2;
        cfg_.vocab_size = 4;
        cfg_.blank_id = 3;
        cfg_.max_symbols_per_step = 3;
        cfg_.durations = {0, 1, 2};
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "c"};
    }
    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }
    struct State : RnntStreamState {};
    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        return std::make_unique<State>();
    }
    void predict_rnnt(RnntStreamState&, int prev_token, int) override {
        last_prev_token = prev_token;
        ++predictor_calls;
    }
    void joint_argmax(RnntStreamState&, const float*, int, int, int32_t*, const float*) override {
        throw std::runtime_error("TDT mock does not implement RNNT argmax");
    }
    void joint_tdt_argmax(
        RnntStreamState&, const float*, int joint_dim, int T, int32_t* tokens,
        int32_t* durations) override {
        check(joint_dim == cfg_.joint_dim && T == 1, "tdt: scalar joint shape");
        ++joint_calls;
        const auto next = rounds.empty() ? std::pair<int, int>{cfg_.blank_id, 1} : rounds.front();
        if (!rounds.empty())
            rounds.pop_front();
        tokens[0] = next.first;
        durations[0] = next.second;
    }

    std::deque<std::pair<int, int>> rounds;
    int predictor_calls = 0;
    int joint_calls = 0;
    int last_prev_token = -1;

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
};

void
test_tdt_duration_and_state_commit() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // t=0 emits a with duration 0, then blank advances to t=1. A duration-2
    // blank skips to t=3, where b emits with duration 2 and finishes T=5.
    eng.rounds = {{0, 0}, {blank, 1}, {blank, 2}, {1, 2}};
    std::vector<float> enc(5 * 2, 0.0f);
    const auto ids = dec.step(enc.data(), 2, 5, 10);
    dec.finalize();
    check(ids == std::vector<int>({0, 1}), "tdt: token and duration sequence");
    check(eng.predictor_calls == 2, "tdt: predictor commits only on non-blank");
    check(dec.last_emit_frame() == 13, "tdt: duration advances encoder frame");
    const auto& words = dec.word_timings();
    check(
        words.size() == 2 && words[0].start_frame == 10 && words[0].end_frame == 11 &&
            words[1].start_frame == 13 && words[1].end_frame == 15,
        "tdt: token timestamps include predicted span");
}

void
test_tdt_zero_duration_guard() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    eng.rounds = {{blank, 0}};
    std::vector<float> enc(2, 0.0f);
    const auto ids = dec.step(enc.data(), 2, 1, 0);
    check(ids.empty() && eng.joint_calls == 1, "tdt: blank duration zero advances immediately");
}

void
test_tdt_duration_crosses_step_boundary() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> frame(2, 0.0f);
    eng.rounds = {{blank, 2}};
    check(dec.step(frame.data(), 2, 1, 0).empty(), "tdt: boundary setup is blank");

    const int calls_before_skip = eng.joint_calls;
    check(dec.step(frame.data(), 2, 1, 1).empty(), "tdt: carried duration skips a whole block");
    check(eng.joint_calls == calls_before_skip, "tdt: skipped block performs no joint call");

    eng.rounds = {{0, 1}};
    const auto ids = dec.step(frame.data(), 2, 1, 2);
    check(ids == std::vector<int>({0}), "tdt: emits after carried duration");
    check(dec.last_emit_frame() == 2, "tdt: carries duration across step boundaries");
}

// ---------------------------------------------------------------------------
// decoder.blank_penalty. The joint already accepts a dense per-vocabulary
// logit bias (added to the logits immediately before the argmax, see
// rnnt_modules.cpp build_joint_tail); the penalty is that bias with
// -blank_penalty at blank_id. These checks pin the contract rather than the
// decoded text, because the mock records the bias instead of applying it.
// ---------------------------------------------------------------------------
void
test_rnnt_blank_penalty() {
    const std::vector<float> enc(4 * 3, 0.0f);  // 3 frames at joint_dim 4

    // Disabled by default: nullptr must still be passed, so the joint graph key
    // (which is keyed on whether a bias is present) is unchanged for every
    // existing deployment.
    {
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng);
        dec.step(enc.data(), 4, 3, 0);
        check(
            eng.bias_log.size() == 1 && eng.bias_log[0].empty(),
            "blank_penalty: default config passes no logit bias");
    }

    // Explicit zero is the same as absent -- not a vector of zeros, which would
    // needlessly take the biased graph path.
    {
        MockRnntEngine eng;
        DecoderConfig cfg;
        cfg.blank_penalty = 0.0;
        RnntGreedyDecoder dec(&eng, cfg);
        dec.step(enc.data(), 4, 3, 0);
        check(
            eng.bias_log.size() == 1 && eng.bias_log[0].empty(),
            "blank_penalty: 0.0 passes no logit bias");
    }

    // Enabled: exactly one non-zero entry, -penalty, at blank_id.
    {
        MockRnntEngine eng;
        const int blank = eng.rnnt_config().blank_id;
        const int vocab = eng.rnnt_config().vocab_size;
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);
        dec.step(enc.data(), 4, 3, 0);

        const std::vector<float>& bias = eng.last_logit_bias;
        const bool sized = static_cast<int>(bias.size()) == vocab;
        check(sized, "blank_penalty: bias is one float per vocabulary entry");
        check(sized && near(bias[blank], -1.5f), "blank_penalty: blank logit biased by -penalty");
        bool rest_zero = sized;
        for (int i = 0; sized && i < vocab; ++i) {
            if (i != blank && bias[i] != 0.0f)
                rest_zero = false;
        }
        check(rest_zero, "blank_penalty: no non-blank token is biased");
    }

    // The bias is not consumed by the first joint call. An emission invalidates
    // the predictor and starts a fresh round at the same frame; that round must
    // carry the penalty too, or the bias would silently apply only to the
    // leading blank run.
    {
        MockRnntEngine eng;
        const int blank = eng.rnnt_config().blank_id;
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);
        eng.rounds = {{0}};  // emit token 0 at frame 0, then an all-blank round
        const auto emitted = dec.step(enc.data(), 4, 3, 0);
        check(emitted == std::vector<int>({0}), "blank_penalty: scripted emission still decodes");
        bool every_round_biased = eng.bias_log.size() >= 2;
        for (const auto& b : eng.bias_log) {
            if (b.size() != static_cast<size_t>(eng.rnnt_config().vocab_size) ||
                !near(b[static_cast<size_t>(blank)], -1.5f))
                every_round_biased = false;
        }
        check(every_round_biased, "blank_penalty: every joint round after an emission is biased");
    }

    // Composition with context biasing. Boosting runs two passes: an unbiased
    // one that decides blank vs non-blank, then a blank-excluded one that only
    // re-ranks the non-blank labels (NeMo/riva rnnt_label_looping semantics).
    // The penalty belongs to the first; the second keeps blank at -1e30.
    {
        MockRnntEngine eng;
        const int blank = eng.rnnt_config().blank_id;
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);

        AsrRequestOptions opts;
        AsrRequestOptions::Boost boost;
        boost.phrases = {"prabhsimran"};
        boost.boost = 5.0f;
        opts.speech_contexts = {boost};
        dec.set_request_options(opts);

        dec.step(enc.data(), 4, 3, 0);
        check(eng.bias_log.size() == 2, "blank_penalty: boosting still runs two joint passes");
        const bool ok = eng.bias_log.size() == 2 &&
                        near(eng.bias_log[0][static_cast<size_t>(blank)], -1.5f) &&
                        eng.bias_log[1][static_cast<size_t>(blank)] < -1e29f;
        check(ok, "blank_penalty: penalty on the blank-vs-non-blank pass, exclusion on the re-rank");
    }

    // TDT: the joint tail throws on a logit bias, so the constructor must
    // disable the penalty rather than let a decode fail at the first frame.
    {
        MockRnntEngine eng;
        eng.set_durations({0, 1, 2, 4});
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);
        dec.step(enc.data(), 4, 3, 0);
        check(
            !eng.bias_log.empty() && eng.bias_log[0].empty(),
            "blank_penalty: ignored for a TDT config instead of reaching the joint");
    }
}

// Per-request blank_penalty (AsrRequestOptions::blank_penalty). The startup key
// above fixes a penalty for the process; this one lets a single decoder serve
// several penalties across requests, so one loaded model can be re-decoded at a
// different penalty instead of hosting a second copy of the weights.
//
// Sentinel: negative inherits the configured value. Zero cannot be the sentinel
// because zero is a real setting -- unbiased decoding -- so these checks pin
// that an explicit 0 request overrides a non-zero config rather than being
// read as "unset".
void
test_rnnt_blank_penalty_per_request() {
    const std::vector<float> enc(4 * 3, 0.0f);  // 3 frames at joint_dim 4

    // A request penalty reaches the joint on an otherwise default decoder.
    {
        MockRnntEngine eng;
        const int blank = eng.rnnt_config().blank_id;
        RnntGreedyDecoder dec(&eng);
        AsrRequestOptions opts;
        opts.blank_penalty = 2.0;
        dec.set_request_options(opts);
        dec.step(enc.data(), 4, 3, 0);
        const std::vector<float>& bias = eng.last_logit_bias;
        check(
            static_cast<int>(bias.size()) == eng.rnnt_config().vocab_size &&
                near(bias[blank], -2.0f),
            "blank_penalty/request: request value reaches the joint");
    }

    // Default options (negative) inherit the configured penalty rather than
    // silently disabling it, which would change behaviour for every existing
    // caller that sets the startup key and sends no override.
    {
        MockRnntEngine eng;
        const int blank = eng.rnnt_config().blank_id;
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);
        dec.set_request_options(AsrRequestOptions{});
        dec.step(enc.data(), 4, 3, 0);
        check(
            !eng.last_logit_bias.empty() && near(eng.last_logit_bias[blank], -1.5f),
            "blank_penalty/request: unset request inherits the configured penalty");
    }

    // Explicit 0 overrides a non-zero config and must take the nullptr-bias
    // path, not pass a vector of zeros.
    {
        MockRnntEngine eng;
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);
        AsrRequestOptions opts;
        opts.blank_penalty = 0.0;
        dec.set_request_options(opts);
        dec.step(enc.data(), 4, 3, 0);
        check(
            eng.bias_log.size() == 1 && eng.bias_log[0].empty(),
            "blank_penalty/request: explicit 0 overrides the config and passes no bias");
    }

    // Successive requests on one decoder must not contaminate each other: the
    // whole point is re-decoding the same audio at a different penalty, so a
    // stale bias would silently corrupt the second attempt.
    {
        MockRnntEngine eng;
        const int blank = eng.rnnt_config().blank_id;
        DecoderConfig cfg;
        cfg.blank_penalty = 1.5;
        RnntGreedyDecoder dec(&eng, cfg);

        AsrRequestOptions first;
        first.blank_penalty = 2.0;
        dec.set_request_options(first);
        dec.step(enc.data(), 4, 3, 0);
        const bool first_ok =
            !eng.last_logit_bias.empty() && near(eng.last_logit_bias[blank], -2.0f);

        // Back to an un-overridden request: restores the configured value, not
        // the previous request's.
        dec.reset();
        dec.set_request_options(AsrRequestOptions{});
        dec.step(enc.data(), 4, 3, 0);
        const bool second_ok =
            !eng.last_logit_bias.empty() && near(eng.last_logit_bias[blank], -1.5f);

        check(first_ok, "blank_penalty/request: first request uses its override");
        check(
            second_ok,
            "blank_penalty/request: later request restores the config, not the previous override");
    }

    // A request penalty of 0 on a default (0) config stays on the unbiased path.
    {
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng);
        AsrRequestOptions opts;
        opts.blank_penalty = 0.0;
        dec.set_request_options(opts);
        dec.step(enc.data(), 4, 3, 0);
        check(
            eng.bias_log.size() == 1 && eng.bias_log[0].empty(),
            "blank_penalty/request: 0 on a default config passes no bias");
    }

    // TDT rejects a logit bias however the penalty arrives.
    {
        MockRnntEngine eng;
        eng.set_durations({0, 1, 2, 4});
        RnntGreedyDecoder dec(&eng);
        AsrRequestOptions opts;
        opts.blank_penalty = 1.5;
        dec.set_request_options(opts);
        dec.step(enc.data(), 4, 3, 0);
        check(
            !eng.bias_log.empty() && eng.bias_log[0].empty(),
            "blank_penalty/request: ignored for TDT even when set per request");
    }
}

}  // namespace

int
main() {
    test_sentencepiece_punctuation_detokenization();
    test_greedy_collapse_and_blank();
    test_greedy_cross_chunk_collapse();
    test_greedy_seam_dedup();
    test_greedy_last_emit_frame();
    test_greedy_confidence();
    test_greedy_word_timings();
    test_greedy_punctuation_word_timing();
    test_greedy_reset_utterance_vs_reset();
    test_greedy_open_word_needs_finalize();
    test_rnnt_emission_and_blank();
    test_rnnt_max_symbols_cap();
    test_rnnt_blank_penalty();
    test_rnnt_blank_penalty_per_request();
    test_rnnt_finalization_preserves_native_punctuation();
    test_rnnt_stalls_do_not_inject_punctuation();
    test_rnnt_late_punctuation_returns_to_previous_final();
    test_rnnt_endpoint_punctuation_continuity();
    test_rnnt_state_threading_and_reset_utterance();
    test_rnnt_word_timings();
    test_rnnt_punctuation_word_timing();
    test_rnnt_open_word_needs_finalize();
    test_tdt_duration_and_state_commit();
    test_tdt_zero_duration_guard();
    test_tdt_duration_crosses_step_boundary();
    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
