// Wraps the ONNX graphs exported by vision2words.inference.export_onnx and
// implements greedy and beam-search decoding for decoder A (attention-LSTM,
// via explicit carried state) and decoder B (Transformer, via the
// fixed-length padded buffer -- see export_onnx.py for why each decoder
// uses a different export strategy).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace v2w {

constexpr int kFeatureDim = 1280;
constexpr int kSpatialTokens = 49;
constexpr int kFixedDecodeLen = 20;  // must match export_onnx.FIXED_DECODE_LEN

struct DecodeResult {
    std::vector<int64_t> tokens;  // includes the leading <START>
    double seconds = 0.0;
};

class CaptionModel {
public:
    CaptionModel(const std::string& encoder_path,
                 const std::string& lstm_attention_init_path,
                 const std::string& lstm_attention_step_path,
                 const std::string& transformer_path);

    // image_chw: 3*224*224 float32, see preprocess.hpp. Returns the 49x1280
    // spatial feature grid (decoders A and B both read spatial features).
    std::vector<float> run_encoder(const std::vector<float>& image_chw);

    DecodeResult decode_attention_greedy(const std::vector<float>& features, int start_id, int end_id, int max_len);
    DecodeResult decode_attention_beam(const std::vector<float>& features, int start_id, int end_id, int beam_size, int max_len);

    DecodeResult decode_transformer_greedy(const std::vector<float>& features, int start_id, int end_id, int pad_id);
    DecodeResult decode_transformer_beam(const std::vector<float>& features, int start_id, int end_id, int pad_id, int beam_size);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;

    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> attn_init_session_;
    std::unique_ptr<Ort::Session> attn_step_session_;
    std::unique_ptr<Ort::Session> transformer_session_;

    // Runs attn_init_session_: features -> (h0, c0), each hidden_size floats.
    void attention_init(const std::vector<float>& features, std::vector<float>& h, std::vector<float>& c);
    // Runs attn_step_session_: (token, h, c, features) -> (logits, h, c, attn_weights).
    std::vector<float> attention_step(int64_t token, std::vector<float>& h, std::vector<float>& c,
                                       const std::vector<float>& features);
    // Runs transformer_session_ on a fixed-length token buffer, returns logits (kFixedDecodeLen * vocab_size).
    std::vector<float> transformer_forward(const std::vector<int64_t>& token_buffer, const std::vector<float>& features);
};

}  // namespace v2w
