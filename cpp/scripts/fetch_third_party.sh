#!/usr/bin/env bash
# Fetches the C++ build's header-only dependencies and the Linux ONNX
# Runtime prebuilt package into cpp/third_party/. Not committed to git --
# see cpp/third_party in .gitignore -- because the ONNX Runtime binary is
# platform-specific; run fetch_third_party.ps1 instead on Windows.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
third_party="$root/cpp/third_party"
ort_version="1.20.1"

mkdir -p "$third_party/stb" "$third_party/nlohmann"

curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -o "$third_party/stb/stb_image.h"
curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h" -o "$third_party/stb/stb_image_resize2.h"
curl -fsSL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" -o "$third_party/nlohmann/json.hpp"

arch="$(uname -m)"
if [ "$arch" = "aarch64" ]; then ort_arch="aarch64"; else ort_arch="x64"; fi

ort_tgz="$third_party/onnxruntime.tgz"
curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/onnxruntime-linux-${ort_arch}-${ort_version}.tgz" -o "$ort_tgz"
tar -xzf "$ort_tgz" -C "$third_party"
rm "$ort_tgz"

echo "Fetched stb, nlohmann/json, and onnxruntime-linux-${ort_arch}-${ort_version} into $third_party"
