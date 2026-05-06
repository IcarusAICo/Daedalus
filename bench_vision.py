"""Benchmark vision models on Sudoku grid extraction.

Sends the same screenshot to each model 5 times and compares to ground truth.
"""

import base64
import os
import sys
import time
from pathlib import Path

import litellm

IMAGE_PATH = Path("traces/t_20260505_072733_1610504b/screens/0009.png")

GROUND_TRUTH = """\
000000003
008601020
000300400
070000050
901200000
400806030
054700000
100000060
006004700"""

PROMPT = (
    "Read the 9x9 Sudoku grid from this screenshot. "
    "Output ONLY the 9 rows, each as 9 digits with 0 for empty cells. "
    "No other text, no spaces, no separators — just 9 lines of 9 digits."
)

MODELS = {
    "opus-4.6": "bedrock/us.anthropic.claude-opus-4-6-v1",
    "sonnet-4.6": "bedrock/us.anthropic.claude-sonnet-4-6",
    "pixtral-large": "bedrock/us.mistral.pixtral-large-2502-v1:0",
}

TRIALS = 5


def make_messages(image_b64: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    ]


def normalize(text: str) -> str:
    """Extract just 81 digits from the response."""
    lines = []
    for line in text.strip().splitlines():
        digits = "".join(c for c in line if c.isdigit())
        if len(digits) == 9:
            lines.append(digits)
    if len(lines) == 9:
        return "\n".join(lines)
    # Fallback: just extract all digits
    all_digits = "".join(c for c in text if c.isdigit())
    if len(all_digits) == 81:
        return "\n".join(all_digits[i:i+9] for i in range(0, 81, 9))
    return text.strip()


def diff_count(response: str, truth: str) -> int:
    """Count differing cells between response and ground truth."""
    r_digits = "".join(c for c in response if c.isdigit())
    t_digits = "".join(c for c in truth if c.isdigit())
    if len(r_digits) != 81 or len(t_digits) != 81:
        return -1  # unparseable
    return sum(1 for a, b in zip(r_digits, t_digits) if a != b)


def main():
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    png = IMAGE_PATH.read_bytes()
    image_b64 = base64.b64encode(png).decode("ascii")
    messages = make_messages(image_b64)

    gt_flat = "".join(c for c in GROUND_TRUTH if c.isdigit())
    print(f"Ground truth ({len(gt_flat)} digits):")
    print(GROUND_TRUTH)
    print("=" * 60)

    results = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({model_id})")
        print("=" * 60)
        model_results = []

        for trial in range(TRIALS):
            print(f"\n  Trial {trial + 1}/{TRIALS}...", end=" ", flush=True)
            t0 = time.time()
            try:
                kwargs = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.0,
                    "timeout": 120,
                    "num_retries": 1,
                }

                raw = litellm.completion(**kwargs)
                content = raw.choices[0].message.content or ""
                dur = time.time() - t0

                normalized = normalize(content)
                errors = diff_count(normalized, GROUND_TRUTH)

                print(f"({dur:.1f}s, {errors} errors)")
                if errors != 0:
                    print(f"    Got: {normalized.replace(chr(10), ' | ')}")

                model_results.append({
                    "trial": trial + 1,
                    "errors": errors,
                    "duration": dur,
                    "raw": content[:200],
                    "normalized": normalized,
                })
            except Exception as exc:
                dur = time.time() - t0
                print(f"FAILED ({dur:.1f}s): {exc}")
                model_results.append({
                    "trial": trial + 1,
                    "errors": -1,
                    "duration": dur,
                    "raw": str(exc)[:200],
                    "normalized": "",
                })

        results[model_name] = model_results

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<16} {'Perfect':>8} {'Avg Errors':>11} {'Avg Time':>9}")
    print("-" * 50)
    for model_name, trials in results.items():
        valid = [t for t in trials if t["errors"] >= 0]
        perfect = sum(1 for t in valid if t["errors"] == 0)
        avg_err = sum(t["errors"] for t in valid) / len(valid) if valid else -1
        avg_time = sum(t["duration"] for t in trials) / len(trials)
        failed = len(trials) - len(valid)
        extra = f" ({failed} failed)" if failed else ""
        print(f"{model_name:<16} {perfect}/{len(valid):>6} {avg_err:>9.1f} {avg_time:>8.1f}s{extra}")


if __name__ == "__main__":
    main()
