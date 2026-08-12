# Ioannis Kalaitzidis, MTE25012

"""
cross_model.py
==============
Προαιρετικό task: συγκριτική αξιολόγηση μεταξύ μοντέλων.

Τρέχει και τον baseline και τον θωρακισμένο βοηθό κάτω από δύο διαφορετικά
backend μοντέλων και απαντά στο ερώτημα που τονίζει η εκφώνηση: ΜΕΤΑΦΕΡΟΝΤΑΙ οι
άμυνες μεταξύ μοντέλων, ή είναι κουρδισμένες για ένα συγκεκριμένο;

Για κάθε μοντέλο αναφέρουμε baseline έναντι θωρακισμένου ως προς την επιτυχία
επίθεσης (συνολικά και ανά οικογένεια), και μια διαφορά "μεταφοράς άμυνας" =
baseline_attack - defended_attack. Μια άμυνα που γενικεύει θα δίνει παρόμοια
θετική διαφορά και στα δύο μοντέλα.

Χρήση:
    # offline (εκφυλισμένο: και τα δύο είναι το ίδιο replay backend)
    python3 cross_model.py --models replay replay

    # live (η πραγματική σύγκριση της εργασίας: Llama έναντι Mistral)
    python3 cross_model.py --models ollama:llama3.1:8b ollama:mistral

Σημείωση offline: με replay και στις δύο πλευρές οι δύο στήλες είναι ταυτόσημες και
δεν διαρρέει τίποτα, οπότε οι διαφορές είναι ~0. Τα πραγματικά νούμερα απαιτούν δύο
live providers.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import config
from providers import get_provider
from runner import run_case, compute_summary


def _load_cases() -> List[Dict[str, Any]]:
    # Φόρτωση του corpus (καλόπιστες + ανταγωνιστικές) σε μία λίστα.
    corpus = json.loads(config.CORPUS_JSON.read_text(encoding="utf-8"))
    return corpus["benign_prompts"] + corpus["adversarial_prompts"]


def _run(system: str, provider_name: str, cases) -> Dict[str, Any]:
    # Τρέχει όλο το benchmark για ένα σύστημα (baseline ή defended) με συγκεκριμένο provider.
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
    # Για ένα μοντέλο: τρέχει baseline και defended και υπολογίζει τη "μεταφορά άμυνας".
    base = _run("baseline", provider_name, cases)
    def_ = _run("defended", provider_name, cases)
    # transfer = πόσο μειώθηκε η επιτυχία επίθεσης χάρη στις άμυνες, για αυτό το μοντέλο.
    transfer = round(base["attack_success_rate"] - def_["attack_success_rate"], 4)
    # Το ίδιο, αναλυμένο ανά οικογένεια.
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
    # Διαβάζει τα δύο μοντέλα από τη γραμμή εντολών, τρέχει τη σύγκριση και αποθηκεύει.
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs=2, default=["replay", "replay"],
                        help="δύο provider names, π.χ. ollama:llama3.1:8b ollama:mistral")
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
