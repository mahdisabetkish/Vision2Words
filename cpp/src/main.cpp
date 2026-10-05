// v2w_caption: caption a single image with the exported encoder + decoder A
// (attention-LSTM) or decoder B (Transformer), via ONNX Runtime.
//
// Usage:
//   v2w_caption <image> --model-dir <export dir> --decoder lstm_attention|transformer
//               [--beam-size N] [--benchmark N]
//
// --model-dir must contain encoder.onnx, vocab.json, and either
// (lstm_attention_init.onnx + lstm_attention_step.onnx) or transformer.onnx,
// as written by `v2w export-onnx`.
#include <chrono>
#include <iostream>
#include <optional>
#include <string>

#include "caption_model.hpp"
#include "preprocess.hpp"
#include "vocabulary.hpp"

namespace {

struct Args {
    std::string image_path;
    std::string model_dir;
    std::string decoder = "lstm_attention";
    int beam_size = 1;
    int benchmark_runs = 0;
};

std::optional<Args> parse_args(int argc, char** argv) {
    if (argc < 2) return std::nullopt;
    Args args;
    args.image_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--model-dir") args.model_dir = next();
        else if (flag == "--decoder") args.decoder = next();
        else if (flag == "--beam-size") args.beam_size = std::stoi(next());
        else if (flag == "--benchmark") args.benchmark_runs = std::stoi(next());
        else throw std::runtime_error("unknown flag: " + flag);
    }
    if (args.model_dir.empty()) return std::nullopt;
    return args;
}

void print_usage() {
    std::cerr << "usage: v2w_caption <image> --model-dir <export dir> "
                 "--decoder lstm_attention|transformer [--beam-size N] [--benchmark N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    auto parsed = parse_args(argc, argv);
    if (!parsed) {
        print_usage();
        return 1;
    }
    Args args = *parsed;

    try {
        v2w::Vocabulary vocab = v2w::Vocabulary::load(args.model_dir + "/vocab.json");
        v2w::CaptionModel model(
            args.model_dir + "/encoder.onnx",
            args.model_dir + "/lstm_attention_init.onnx",
            args.model_dir + "/lstm_attention_step.onnx",
            args.model_dir + "/transformer.onnx"
        );

        auto caption_once = [&]() -> v2w::DecodeResult {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<float> image = v2w::load_and_preprocess(args.image_path);
            std::vector<float> features = model.run_encoder(image);

            v2w::DecodeResult result;
            if (args.decoder == "lstm_attention") {
                result = (args.beam_size > 1)
                    ? model.decode_attention_beam(features, vocab.start_id(), vocab.end_id(), args.beam_size, 20)
                    : model.decode_attention_greedy(features, vocab.start_id(), vocab.end_id(), 20);
            } else if (args.decoder == "transformer") {
                result = (args.beam_size > 1)
                    ? model.decode_transformer_beam(features, vocab.start_id(), vocab.end_id(), vocab.pad_id(), args.beam_size)
                    : model.decode_transformer_greedy(features, vocab.start_id(), vocab.end_id(), vocab.pad_id());
            } else {
                throw std::runtime_error("unknown decoder: " + args.decoder + " (expected lstm_attention or transformer)");
            }
            result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return result;
        };

        if (args.benchmark_runs > 0) {
            // Warm-up run (first call pays for session lazy-init / allocator warm-up).
            caption_once();
            double total_seconds = 0.0;
            for (int i = 0; i < args.benchmark_runs; ++i) {
                total_seconds += caption_once().seconds;
            }
            double mean_ms = 1000.0 * total_seconds / args.benchmark_runs;
            std::cout << "mean end-to-end latency over " << args.benchmark_runs
                      << " runs: " << mean_ms << " ms\n";
            return 0;
        }

        v2w::DecodeResult result = caption_once();
        std::cout << vocab.decode(result.tokens) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
