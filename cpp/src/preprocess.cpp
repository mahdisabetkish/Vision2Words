#include "preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

namespace v2w {

std::vector<float> load_and_preprocess(const std::string& image_path) {
    int width, height, channels;
    unsigned char* pixels = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!pixels) {
        throw std::runtime_error("failed to load image: " + image_path + " (" + stbi_failure_reason() + ")");
    }

    std::vector<unsigned char> resized(kImageSize * kImageSize * 3);
    stbir_resize_uint8_linear(
        pixels, width, height, 0,
        resized.data(), kImageSize, kImageSize, 0,
        STBIR_RGB
    );
    stbi_image_free(pixels);

    // HWC uint8 -> CHW float32, scaled to [0,1] then ImageNet-normalized,
    // matching torchvision's ToTensor() + Normalize(mean, std).
    std::vector<float> chw(3 * kImageSize * kImageSize);
    const int plane = kImageSize * kImageSize;
    for (int y = 0; y < kImageSize; ++y) {
        for (int x = 0; x < kImageSize; ++x) {
            const int pixel_index = (y * kImageSize + x) * 3;
            for (int c = 0; c < 3; ++c) {
                float value = resized[pixel_index + c] / 255.0f;
                value = (value - kImageNetMean[c]) / kImageNetStd[c];
                chw[c * plane + y * kImageSize + x] = value;
            }
        }
    }
    return chw;
}

}  // namespace v2w
