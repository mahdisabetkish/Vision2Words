#include "caption_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace v2w {
namespace {

// Ort::Session wants a wchar_t path on Windows and a char path elsewhere.
#ifdef _WIN32
std::wstring to_ort_path(const std::string& path) {
    return std::wstring(path.begin(), path.end());
}
#else
std::string to_ort_path(const std::string& path) { return path; }
#endif

std::unique_ptr<Ort::Session> make_session(Ort::Env& env, const Ort::SessionOptions& options, const std::string& path) {
    return std::make_unique<Ort::Session>(env, to_ort_path(path).c_str(), options);
}

float log_softmax_max_index(const std::vector<float>& logits, int64_t vocab_size, int64_t& argmax_out) {
    float max_logit = *std::max_element(logits.begin(), logits.begin() + vocab_size);
    argmax_out = static_cast<int64_t>(std::max_element(logits.begin(), logits.begin() + vocab_size) - logits.begin());
    return max_logit;
}

}  // namespace

CaptionModel::CaptionModel(const std::string& encoder_path,
                            const std::string& lstm_attention_init_path,
                            const std::string& lstm_attention_step_path,
                            const std::string& transformer_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "vision2words"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    encoder_session_ = make_session(env_, session_options_, encoder_path);
    attn_init_session_ = make_session(env_, session_options_, lstm_attention_init_path);
    attn_step_session_ = make_session(env_, session_options_, lstm_attention_step_path);
    transformer_session_ = make_session(env_, session_options_, transformer_path);
}

std::vector<float> CaptionModel::run_encoder(const std::vector<float>& image_chw) {
    std::array<int64_t, 4> input_shape{1, 3, kImageSize, kImageSize};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(image_chw.data()), image_chw.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"image"};
    const char* output_names[] = {"pooled", "spatial"};
    auto outputs = encoder_session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

    float* spatial_data = outputs[1].GetTensorMutableData<float>();
    size_t spatial_size = kSpatialTokens * kFeatureDim;
    return std::vector<float>(spatial_data, spatial_data + spatial_size);
}

void CaptionModel::attention_init(const std::vector<float>& features, std::vector<float>& h, std::vector<float>& c) {
    std::array<int64_t, 3> feat_shape{1, kSpatialTokens, kFeatureDim};
    Ort::Value feat_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(features.data()), features.size(), feat_shape.data(), feat_shape.size());

    const char* input_names[] = {"features"};
    const char* output_names[] = {"h0", "c0"};
    auto outputs = attn_init_session_->Run(Ort::RunOptions{nullptr}, input_names, &feat_tensor, 1, output_names, 2);

    auto h_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t hidden_size = static_cast<size_t>(h_shape[1]);
    float* h_data = outputs[0].GetTensorMutableData<float>();
    float* c_data = outputs[1].GetTensorMutableData<float>();
    h.assign(h_data, h_data + hidden_size);
    c.assign(c_data, c_data + hidden_size);
}

std::vector<float> CaptionModel::attention_step(int64_t token, std::vector<float>& h, std::vector<float>& c,
                                                 const std::vector<float>& features) {
    size_t hidden_size = h.size();
    std::array<int64_t, 1> token_shape{1};
    std::array<int64_t, 2> hc_shape{1, static_cast<int64_t>(hidden_size)};
    std::array<int64_t, 3> feat_shape{1, kSpatialTokens, kFeatureDim};

    Ort::Value token_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, &token, 1, token_shape.data(), token_shape.size());
    Ort::Value h_tensor = Ort::Value::CreateTensor<float>(memory_info_, h.data(), h.size(), hc_shape.data(), hc_shape.size());
    Ort::Value c_tensor = Ort::Value::CreateTensor<float>(memory_info_, c.data(), c.size(), hc_shape.data(), hc_shape.size());
    Ort::Value feat_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(features.data()), features.size(), feat_shape.data(), feat_shape.size());

    std::array<Ort::Value, 4> inputs{std::move(token_tensor), std::move(h_tensor), std::move(c_tensor), std::move(feat_tensor)};
    const char* input_names[] = {"token", "h", "c", "features"};
    const char* output_names[] = {"logits", "h_next", "c_next", "attn_weights"};
    auto outputs = attn_step_session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 4);

    float* logits_data = outputs[0].GetTensorMutableData<float>();
    size_t vocab_size = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    std::vector<float> logits(logits_data, logits_data + vocab_size);

    float* h_next = outputs[1].GetTensorMutableData<float>();
    float* c_next = outputs[2].GetTensorMutableData<float>();
    h.assign(h_next, h_next + hidden_size);
    c.assign(c_next, c_next + hidden_size);

    return logits;
}

DecodeResult CaptionModel::decode_attention_greedy(const std::vector<float>& features, int start_id, int end_id, int max_len) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> h, c;
    attention_init(features, h, c);

    DecodeResult result;
    result.tokens.push_back(start_id);
    int64_t current = start_id;
    for (int step = 0; step < max_len; ++step) {
        std::vector<float> logits = attention_step(current, h, c, features);
        int64_t next_token;
        log_softmax_max_index(logits, static_cast<int64_t>(logits.size()), next_token);
        result.tokens.push_back(next_token);
        if (next_token == end_id) break;
        current = next_token;
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

DecodeResult CaptionModel::decode_attention_beam(const std::vector<float>& features, int start_id, int end_id,
                                                  int beam_size, int max_len) {
    auto t0 = std::chrono::steady_clock::now();

    struct Beam {
        std::vector<int64_t> tokens;
        std::vector<float> h, c;
        float score = 0.0f;
        bool finished = false;
    };

    Beam initial;
    initial.tokens.push_back(start_id);
    attention_init(features, initial.h, initial.c);
    std::vector<Beam> beams{initial};
    std::vector<Beam> finished_beams;

    for (int step = 0; step < max_len && !beams.empty(); ++step) {
        std::vector<Beam> candidates;
        for (auto& beam : beams) {
            std::vector<float> h = beam.h, c = beam.c;
            std::vector<float> logits = attention_step(beam.tokens.back(), h, c, features);

            float max_logit = *std::max_element(logits.begin(), logits.end());
            double sum_exp = 0.0;
            for (float v : logits) sum_exp += std::exp(static_cast<double>(v - max_logit));
            double log_sum_exp = std::log(sum_exp) + max_logit;

            std::vector<int> indices(logits.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + beam_size, indices.end(),
                               [&](int a, int b) { return logits[a] > logits[b]; });

            for (int k = 0; k < beam_size; ++k) {
                int token_id = indices[k];
                float log_prob = static_cast<float>(logits[token_id] - log_sum_exp);
                Beam next = beam;
                next.h = h;
                next.c = c;
                next.tokens.push_back(token_id);
                next.score = beam.score + log_prob;
                next.finished = (token_id == end_id);
                candidates.push_back(std::move(next));
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Beam& a, const Beam& b) {
            return a.score / a.tokens.size() > b.score / b.tokens.size();
        });
        if (static_cast<int>(candidates.size()) > beam_size) candidates.resize(beam_size);

        beams.clear();
        for (auto& cand : candidates) {
            if (cand.finished) {
                finished_beams.push_back(std::move(cand));
            } else {
                beams.push_back(std::move(cand));
            }
        }
    }

    for (auto& beam : beams) finished_beams.push_back(std::move(beam));
    auto best = std::max_element(finished_beams.begin(), finished_beams.end(), [](const Beam& a, const Beam& b) {
        return a.score / a.tokens.size() < b.score / b.tokens.size();
    });

    DecodeResult result;
    result.tokens = best->tokens;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

std::vector<float> CaptionModel::transformer_forward(const std::vector<int64_t>& token_buffer,
                                                       const std::vector<float>& features) {
    std::array<int64_t, 3> feat_shape{1, kSpatialTokens, kFeatureDim};
    std::array<int64_t, 2> tok_shape{1, kFixedDecodeLen};

    Ort::Value feat_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(features.data()), features.size(), feat_shape.data(), feat_shape.size());
    Ort::Value tok_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t*>(token_buffer.data()), token_buffer.size(), tok_shape.data(), tok_shape.size());

    std::array<Ort::Value, 2> inputs{std::move(feat_tensor), std::move(tok_tensor)};
    const char* input_names[] = {"features", "tokens"};
    const char* output_names[] = {"logits"};
    auto outputs = transformer_session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 1);

    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // (1, kFixedDecodeLen, vocab_size)
    size_t total = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + total);
}

DecodeResult CaptionModel::decode_transformer_greedy(const std::vector<float>& features, int start_id, int end_id, int pad_id) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int64_t> buffer(kFixedDecodeLen, pad_id);
    buffer[0] = start_id;

    DecodeResult result;
    result.tokens.push_back(start_id);
    int generated = 1;
    for (int step = 0; step < kFixedDecodeLen - 1; ++step) {
        std::vector<float> logits = transformer_forward(buffer, features);
        int64_t vocab_size = static_cast<int64_t>(logits.size()) / kFixedDecodeLen;
        const float* step_logits = logits.data() + step * vocab_size;
        auto max_it = std::max_element(step_logits, step_logits + vocab_size);
        int64_t next_token = max_it - step_logits;

        result.tokens.push_back(next_token);
        if (next_token == end_id || generated >= kFixedDecodeLen - 1) break;
        buffer[generated] = next_token;
        generated++;
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

DecodeResult CaptionModel::decode_transformer_beam(const std::vector<float>& features, int start_id, int end_id,
                                                    int pad_id, int beam_size) {
    auto t0 = std::chrono::steady_clock::now();

    struct Beam {
        std::vector<int64_t> tokens{start_id};
        float score = 0.0f;
        bool finished = false;
    };
    std::vector<Beam> beams(1);
    std::vector<Beam> finished_beams;

    for (int step = 0; step < kFixedDecodeLen - 1 && !beams.empty(); ++step) {
        std::vector<Beam> candidates;
        for (auto& beam : beams) {
            std::vector<int64_t> buffer(kFixedDecodeLen, pad_id);
            std::copy(beam.tokens.begin(), beam.tokens.end(), buffer.begin());

            std::vector<float> logits = transformer_forward(buffer, features);
            int64_t vocab_size = static_cast<int64_t>(logits.size()) / kFixedDecodeLen;
            const float* step_logits = logits.data() + step * vocab_size;

            float max_logit = *std::max_element(step_logits, step_logits + vocab_size);
            double sum_exp = 0.0;
            for (int64_t i = 0; i < vocab_size; ++i) sum_exp += std::exp(static_cast<double>(step_logits[i] - max_logit));
            double log_sum_exp = std::log(sum_exp) + max_logit;

            std::vector<int> indices(vocab_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + beam_size, indices.end(),
                               [&](int a, int b) { return step_logits[a] > step_logits[b]; });

            for (int k = 0; k < beam_size; ++k) {
                int token_id = indices[k];
                float log_prob = static_cast<float>(step_logits[token_id] - log_sum_exp);
                Beam next = beam;
                next.tokens.push_back(token_id);
                next.score = beam.score + log_prob;
                next.finished = (token_id == end_id) || (static_cast<int>(next.tokens.size()) >= kFixedDecodeLen);
                candidates.push_back(std::move(next));
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Beam& a, const Beam& b) {
            return a.score / a.tokens.size() > b.score / b.tokens.size();
        });
        if (static_cast<int>(candidates.size()) > beam_size) candidates.resize(beam_size);

        beams.clear();
        for (auto& cand : candidates) {
            if (cand.finished) {
                finished_beams.push_back(std::move(cand));
            } else {
                beams.push_back(std::move(cand));
            }
        }
    }

    for (auto& beam : beams) finished_beams.push_back(std::move(beam));
    auto best = std::max_element(finished_beams.begin(), finished_beams.end(), [](const Beam& a, const Beam& b) {
        return a.score / a.tokens.size() < b.score / b.tokens.size();
    });

    DecodeResult result;
    result.tokens = best->tokens;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

}  // namespace v2w
