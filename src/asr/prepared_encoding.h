// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Immutable capture of one utterance's cache-aware encoder output, so the same
// acoustic representation can be decoded more than once.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nemo_speech::asr {

// What the cache-aware encoder produced for a whole utterance, in the
// joint-projected representation the transducer head consumes.
//
// The point of keeping it is that decoding is cheap relative to encoding, so a
// caller that wants a second hypothesis under different decoder settings -- a
// different blank penalty, say -- should not pay for a second encoder pass over
// the same audio. The representation itself is independent of any decoder
// setting, which is exactly why it can be reused.
//
// Immutable by contract: Recognizer::decode_prepared takes it by const
// reference and every decode builds a fresh predictor state and hypothesis
// buffer, so repeated decodes cannot influence one another.
//
// Host memory, not device memory. The encoder already reads its output back to
// the host for the greedy head, so capturing it costs a copy and no device
// lifetime concerns: the handle stays valid after its stream is closed, and
// outlives any backend buffer it was derived from.
struct PreparedEncoding {
    // Column-major (joint_dim, frame_count): frame_count groups of joint_dim
    // contiguous floats, which is the layout Decoder::step expects.
    std::vector<float> frames;
    int frame_count = 0;
    int joint_dim = 0;

    // Prompt conditioning that produced these frames. A prompt-conditioned
    // model fuses the language prompt inside the encoder, so the choice is
    // already baked into the captured representation and cannot be changed by
    // re-decoding. Recorded so a re-decode can report and postprocess under
    // the same language it was encoded with.
    int prompt_index = -1;
    std::string language_code;

    // Seconds of audio behind these frames, for the Result's audio_processed.
    float audio_processed_sec = 0.0f;

    // The Recognizer that produced this. Decoding against a different one
    // would silently mix a representation with a foreign vocabulary and joint
    // network, so it is rejected rather than diagnosed later as bad output.
    const void* owner = nullptr;

    bool valid() const {
        return frame_count > 0 && joint_dim > 0 &&
               frames.size() >= static_cast<size_t>(frame_count) *
                                    static_cast<size_t>(joint_dim);
    }

    // Host bytes held, for callers that bound how many they keep alive.
    size_t size_bytes() const { return frames.size() * sizeof(float); }
};

}  // namespace nemo_speech::asr
