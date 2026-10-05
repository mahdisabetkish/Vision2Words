# Fetches the C++ build's header-only dependencies and the Windows ONNX
# Runtime prebuilt package into cpp/third_party/. Not committed to git --
# see cpp/third_party in .gitignore -- because the ONNX Runtime binary is
# platform-specific; run fetch_third_party.sh instead on Linux/macOS.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$thirdParty = Join-Path $root "cpp\third_party"
$ortVersion = "1.20.1"

New-Item -ItemType Directory -Force -Path "$thirdParty\stb" | Out-Null
New-Item -ItemType Directory -Force -Path "$thirdParty\nlohmann" | Out-Null

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -OutFile "$thirdParty\stb\stb_image.h"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize2.h" -OutFile "$thirdParty\stb\stb_image_resize2.h"
Invoke-WebRequest -Uri "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" -OutFile "$thirdParty\nlohmann\json.hpp"

$ortZip = "$thirdParty\onnxruntime.zip"
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-$ortVersion.zip" -OutFile $ortZip
Expand-Archive -Path $ortZip -DestinationPath $thirdParty -Force
Remove-Item $ortZip

Write-Host "Fetched stb, nlohmann/json, and onnxruntime-win-x64-$ortVersion into $thirdParty"
