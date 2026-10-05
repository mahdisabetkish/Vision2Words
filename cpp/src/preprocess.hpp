// Mirrors vision2words.models.encoder.EncoderCNN.preprocess(): resize to
// 224x224, scale to [0,1], then ImageNet mean/std normalization, as a CHW
// float32 buffer ready for the encoder's "image" input.
//
// The resize filter here (stb_image_resize2's default) is not bit-identical
// to PIL's bilinear resize -- there is no dependency-free way to match PIL's
// exact resampling kernel in C++ without vendoring PIL's own resampling
// code. In practice this shows up as sub-1% pixel-value differences, not as
// different captions; see cpp/README.md for the actual parity numbers
// measured against the Python pipeline.
#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace v2w {

constexpr int kImageSize = 224;
constexpr std::array<float, 3> kImageNetMean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kImageNetStd = {0.229f, 0.224f, 0.225f};

// Returns a flat CHW (3 * 224 * 224) float32 buffer.
std::vector<float> load_and_preprocess(const std::string& image_path);

}  // namespace v2w
