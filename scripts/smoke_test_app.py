"""Check a running demo end to end: caption one image through its Gradio API
and make sure every decoder answered, the retrieval-augmented one included.

The image publishing workflow runs this against the freshly built container
before anything is pushed. It works just as well against a local
`python app/app.py` or `docker run`:

    python scripts/smoke_test_app.py http://localhost:7860 app/examples/2339106348_2df90aa6a9.jpg
"""

from __future__ import annotations

import argparse
import re
import sys

from gradio_client import Client, handle_file

EXPECTED_CARDS = ("attention-lstm", "transformer", "transformer + retrieval")
CARD = re.compile(
    r'v2w-specimen-label"><span class="v2w-dot"></span>(.*?)</div>\s*'
    r'<div class="v2w-caption-text">(.*?)</div>',
    re.DOTALL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("image")
    parser.add_argument("--beam-size", type=int, default=3)
    args = parser.parse_args()

    client = Client(args.url, verbose=False)
    outputs = client.predict(handle_file(args.image), args.beam_size, api_name="/caption_image")
    captions = {}
    for output in outputs:
        if isinstance(output, str):
            for label, caption in CARD.findall(output):
                captions[label.strip()] = caption.strip()
    for label, caption in captions.items():
        print(f"{label}: {caption}")

    problems = []
    for label in EXPECTED_CARDS:
        caption = captions.get(label)
        if caption is None:
            problems.append(f"no caption card for {label}")
        elif caption == "unavailable" or len(caption.split()) < 3:
            problems.append(f"{label} did not produce a caption ({caption!r})")
    if problems:
        sys.exit("smoke test failed: " + "; ".join(problems))
    print("smoke test passed")


if __name__ == "__main__":
    main()
