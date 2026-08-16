"""
cross_model.py
==============
Optional task: cross-model comparative evaluation.

Runs both the baseline and the hardened assistant under two different model
backends and answers the question the assignment emphasizes: do the defenses
TRANSFER across models, or are they tuned for a specific one?

For each model we report baseline versus hardened in terms of attack success
(overall and per family), and a "defense transfer" difference =
baseline_attack - defended_attack. A defense that generalizes will give a similar
positive difference on both models.

Usage:
    # offline (degenerate: both are the same replay backend)
    python cross_model.py --models replay replay

    # live (the real comparison of the work: Llama versus Mistral)
    python cross_model.py --models ollama:llama3.1:8b ollama:mistral

Offline note: with replay on both sides the two columns are identical and nothing
leaks, so the differences are ~0. The real numbers require two live providers.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import config
from providers import get_provider
from runner import run_case, compute_summary


def _load_cases() -> List[Dict[str, Any]]:
    # Load the corpus (benign + adversarial) into one list.
    corpus = json.loads(config.CORPUS_JSON.read_text(encoding="utf-8"))
    return corpus["benign_prompts"] + corpus["adversarial_prompts"]


def _run(system: str, provider_name: str, cases) -> Dict[str, Any]:
    # Runs the whole benchmark for one system (baseline or defended) with a specific provider.
    provider = get_provider(provider_name)
    if system == "baseline":
        from assistant_baseline import BaselineAssistant
        assistant = BaselineAssistant(provider=provider)
    else:
        from assistant_defended import DefendedAssistant
        assistant = DefendedAssistant(provider=provider)
    results = [run_case(assistant, c) for c in cases]
    return compute_summary(results)


def run_model(provider_name: str, cases) -> Dict[str, Any]:
    # For one model: runs baseline and defended and computes the "defense transfer".
    base = _run("baseline", provider_name, cases)
    def_ = _run("defended", provider_name, cases)
    # transfer = how much the attack success dropped thanks to the defenses, for this model.
    transfer = round(base["attack_success_rate"] - def_["attack_success_rate"], 4)
    # The same, broken down per family.
    fam_transfer = {}
    for fam in base["by_family"]:
        b = base["by_family"][fam]["success_rate"]
        d = def_.get("by_family", {}).get(fam, {}).get("success_rate", 0.0)
        fam_transfer[fam] = round(b - d, 4)
    return {
        "provider": provider_name,
        "model": get_provider(provider_name).model,
        "baseline_attack_success": base["attack_success_rate"],
        "defended_attack_success": def_["attack_success_rate"],
        "defense_transfer_delta": transfer,
        "baseline_benign_success": base["benign_success_rate"],
        "defended_benign_success": def_["benign_success_rate"],
        "defended_false_refusal": def_["false_refusal_rate"],
        "per_family_transfer": fam_transfer,
    }


def print_comparison(models: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON")
    print("=" * 80)
    hdr = f"{'Metric':<34}" + "".join(f"{m['model'][:16]:<18}" for m in models)
    print(hdr); print("-" * 80)

    def row(label, key, pct=True):
        line = f"{label:<34}"
        for m in models:
            v = m[key]
            line += (f"{v:.0%}" if pct else f"{v:+.0%}").ljust(18)
        print(line)

    row("Baseline attack success", "baseline_attack_success")
    row("Defended attack success", "defended_attack_success")
    row("Defense transfer (delta)", "defense_transfer_delta", pct=False)
    row("Defended benign success", "defended_benign_success")
    row("Defended false refusal", "defended_false_refusal")
    print("-" * 80)

    print("\nPer-family defense transfer (baseline - defended attack success):")
    fams = sorted({f for m in models for f in m["per_family_transfer"]})
    print(f"{'Family':<34}" + "".join(f"{m['model'][:16]:<18}" for m in models))
    for fam in fams:
        line = f"{fam:<34}"
        for m in models:
            line += f"{m['per_family_transfer'].get(fam, 0.0):+.0%}".ljust(18)
        print(line)


def main() -> None:
    # Reads the two models from the command line, runs the comparison and stores it.
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs=2, default=["replay", "replay"],
                        help="two provider names, e.g. ollama:llama3.1:8b ollama:mistral")
    args = parser.parse_args()

    config.ensure_dirs()
    if not config.CORPUS_JSON.exists():
        import benchmark_generator
        benchmark_generator.main()

    cases = _load_cases()
    print("=" * 80)
    print(f"CROSS-MODEL STUDY: {args.models[0]} vs {args.models[1]}")
    if args.models[0] == args.models[1] == "replay":
        print("NOTE: both backends are replay (offline) -> identical, degenerate columns.")
    print("=" * 80)

    models = [run_model(name, cases) for name in args.models]
    print_comparison(models)

    out = {"models_compared": args.models, "results": models}
    out_path = config.RESULTS_DIR / "cross_model.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
