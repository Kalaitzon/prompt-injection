"""
ablation.py
===========
task 5

The hardened assistant has three independent defense layers (L1 input
validation, L2 context separation, L3 output filtering). This file tests all 8
enable/disable combinations of the three layers (2^3 = 8) and for each reports
the attack success/block rate, benign success, false refusals, and attack
success per family.

Each combination is executed N_RUNS times and the result is given as a mean
+/- standard deviation, so that the variance is shown and not just a single
number. Under temperature=0 the repetitions come out identical, so the standard
deviation is 0, which we explicitly state as an indication of reproducibility.

Beyond the 8 combinations, the marginal contribution of each layer is also
computed, that is, the mean effect that its activation has when the other two
remain fixed. This shows how much each layer is worth individually, without
relying on a single configuration.

Input: the benchmark corpus (from benchmark_generator.py)
Output: results/ablation.json and tables on the console
"""

from __future__ import annotations

import json
import statistics
from itertools import product
from typing import Any, Dict, List, Tuple

import config
from runner import run_case, compute_summary


# Number of repetitions per configuration. Default 3, but configurable via the
# BANK_ABLATION_RUNS environment variable (e.g. =1 for a fast deterministic run, since
# under temperature=0 each repetition is identical anyway).
N_RUNS = int(__import__("os").environ.get("BANK_ABLATION_RUNS", "3"))

# The 8 layer combinations, keyed on the (L1, L2, L3) on/off triple, mapped to a
# readable label used in the tables and in results/ablation.json.
CONFIG_NAMES = {
    (False, False, False): "None (baseline)",   # no defense -> equivalent to the task 3 baseline
    (True,  False, False): "L1 only (input)",
    (False, True,  False): "L2 only (context)",
    (False, False, True):  "L3 only (output)",
    (True,  True,  False): "L1+L2",
    (True,  False, True):  "L1+L3",
    (False, True,  True):  "L2+L3",
    (True,  True,  True):  "L1+L2+L3 (full)",
}


def _load_cases() -> List[Dict[str, Any]]:
    # Load the corpus and join benign + adversarial cases into one list, so that
    # each configuration is tested on exactly the same 60 cases.
    corpus = json.loads(config.CORPUS_JSON.read_text(encoding="utf-8"))
    return corpus["benign_prompts"] + corpus["adversarial_prompts"]


def _evaluate_once(l1: bool, l2: bool, l3: bool, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs the full benchmark once for a given layer configuration (silently)."""
    # Build a hardened assistant with exactly the requested layers active. This is
    # what makes ablation possible: each layer is enabled independently.
    from assistant_defended import DefendedAssistant
    assistant = DefendedAssistant(enable_layer1=l1, enable_layer2=l2, enable_layer3=l3,
                                  system_label=f"abl_{int(l1)}{int(l2)}{int(l3)}")
    # Run each case through this assistant and summarize into the standard metrics.
    results = [run_case(assistant, c) for c in cases]
    return compute_summary(results)


def run_config(l1: bool, l2: bool, l3: bool, cases: List[Dict[str, Any]],
               n_runs: int = N_RUNS) -> Dict[str, Any]:
    # Run a configuration n_runs times and aggregate. The repetition is what
    # allows us to report variance (std) instead of a single number.
    name = CONFIG_NAMES[(l1, l2, l3)]
    runs = [_evaluate_once(l1, l2, l3, cases) for _ in range(n_runs)]

    def agg(key: str) -> Dict[str, float]:
        # Mean and (population) standard deviation of a metric over the n_runs repetitions.
        vals = [r[key] for r in runs]
        return {
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        }

    # Mean attack success per family: shows which attack type each configuration stops.
    families = sorted({f for r in runs for f in r["by_family"]})
    by_family = {}
    for fam in families:
        rates = [r["by_family"][fam]["success_rate"] for r in runs if fam in r["by_family"]]
        by_family[fam] = {
            "mean": round(statistics.fmean(rates), 4),
            "std": round(statistics.pstdev(rates), 4) if len(rates) > 1 else 0.0,
        }

    return {
        "config": {"l1": l1, "l2": l2, "l3": l3},
        "name": name,
        "n_runs": n_runs,
        "attack_success_rate": agg("attack_success_rate"),
        "attack_block_rate": agg("attack_block_rate"),
        "benign_success_rate": agg("benign_success_rate"),
        "false_refusal_rate": agg("false_refusal_rate"),
        "by_family": by_family,
    }


def marginal_contributions(by_config: Dict[Tuple[bool, bool, bool], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Mean marginal effect of activating each layer, with the other two fixed.
    A positive 'attack_reduction' means a security benefit, a positive
    'refusal_increase' means a usability cost.

    For each layer we look at the 4 contexts given by the on/off states of the
    other two layers, measure how much the metric changes when we turn it on in
    each context, and take the average. This way the value does not depend on a
    single configuration.
    """
    out: Dict[str, Any] = {}
    for layer_idx, layer_name in enumerate(["L1_input", "L2_context", "L3_output"]):
        attack_deltas: List[float] = []
        refusal_deltas: List[float] = []
        # 'others' enumerates the 4 on/off states of the other two layers.
        for others in product([False, True], repeat=2):
            # Build the two configuration keys that differ ONLY in this layer:
            # one with it OFF and one with it ON, with the other two layers identical.
            off_key = list(others); off_key.insert(layer_idx, False); off_key = tuple(off_key)
            on_key = list(others); on_key.insert(layer_idx, True); on_key = tuple(on_key)
            off = by_config[off_key]; on = by_config[on_key]
            # Security benefit: how much the attack success DECREASES when the layer is ON.
            attack_deltas.append(off["attack_success_rate"]["mean"]
                                 - on["attack_success_rate"]["mean"])
            # Usability cost: how much the false refusals INCREASE when the layer is ON.
            refusal_deltas.append(on["false_refusal_rate"]["mean"]
                                  - off["false_refusal_rate"]["mean"])
        out[layer_name] = {
            "mean_attack_reduction": round(statistics.fmean(attack_deltas), 4),
            "mean_refusal_increase": round(statistics.fmean(refusal_deltas), 4),
            "n_contexts": len(attack_deltas),
        }
    return out


def _cell(text: str, width: int) -> str:
    return text.ljust(width)


def print_table(all_configs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("ABLATION - COMPARATIVE RESULTS (mean +/- std over runs)")
    print("=" * 92)
    print(_cell("Configuration", 22) + _cell("AttackSucc", 16) + _cell("Block", 12)
          + _cell("Benign", 16) + _cell("FalseRefuse", 14))
    print("-" * 92)
    for c in all_configs:
        a = c["attack_success_rate"]; b = c["attack_block_rate"]
        bn = c["benign_success_rate"]; fr = c["false_refusal_rate"]
        print(
            _cell(c["name"], 22)
            + _cell(f"{a['mean']:.0%}±{a['std']:.0%}", 16)
            + _cell(f"{b['mean']:.0%}", 12)
            + _cell(f"{bn['mean']:.0%}±{bn['std']:.0%}", 16)
            + _cell(f"{fr['mean']:.0%}", 14)
        )
    print("-" * 92)


def print_family_table(all_configs: List[Dict[str, Any]]) -> None:
    fams = sorted({f for c in all_configs for f in c["by_family"]})
    print("\n" + "=" * 92)
    print("ATTACK SUCCESS BY FAMILY (mean across runs)")
    print("=" * 92)
    header = f"{'Family':<34}" + "".join(c["name"][:10].ljust(11) for c in all_configs)
    print(header)
    print("-" * 92)
    for fam in fams:
        row = f"{fam:<34}"
        for c in all_configs:
            v = c["by_family"].get(fam, {"mean": 0.0})
            row += f"{v['mean']:.0%}".ljust(11)
        print(row)
    print("-" * 92)


def print_marginal(marg: Dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("MARGINAL CONTRIBUTION OF EACH LAYER (averaged over the other two)")
    print("=" * 92)
    print(_cell("Layer", 16) + _cell("Mean attack reduction", 26)
          + _cell("Mean false-refusal cost", 26) + "contexts")
    print("-" * 92)
    for layer, v in marg.items():
        print(_cell(layer, 16)
              + _cell(f"{v['mean_attack_reduction']:+.0%}", 26)
              + _cell(f"{v['mean_refusal_increase']:+.0%}", 26)
              + str(v["n_contexts"]))
    print("-" * 92)


def main() -> None:
    # Ensure the output folders exist, and generate the corpus if missing.
    config.ensure_dirs()
    if not config.CORPUS_JSON.exists():
        import benchmark_generator
        benchmark_generator.main()

    cases = _load_cases()
    keys = list(CONFIG_NAMES.keys())

    print("=" * 92)
    print(f"ABLATION STUDY  (provider via config, n_runs={N_RUNS})")
    print("NOTE: offline replay/mock is deterministic and never leaks canaries, so all")
    print("      configs read 0% attack success here. Real numbers require a live provider.")
    print("=" * 92)

    # Run all 8 configurations, keeping the results both by key (for the marginal-
    # contribution analysis) and in order (for the print/store tables).
    by_config: Dict[Tuple[bool, bool, bool], Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    for (l1, l2, l3) in keys:
        res = run_config(l1, l2, l3, cases)
        by_config[(l1, l2, l3)] = res
        ordered.append(res)
        print(f"  done: {res['name']:<22} "
              f"attack={res['attack_success_rate']['mean']:.0%} "
              f"benign={res['benign_success_rate']['mean']:.0%}")

    # Compute the marginal contribution of each layer from the 8 configurations.
    marg = marginal_contributions(by_config)

    # Print the three readable tables (overall / per family / marginal).
    print_table(ordered)
    print_family_table(ordered)
    print_marginal(marg)

    # Store everything in results/ablation.json (the figures read from this file).
    out = {
        "metadata": {"n_runs": N_RUNS, "provider": config.PROVIDER,
                     "temperature": config.TEMPERATURE},
        "configurations": ordered,
        "marginal_contributions": marg,
    }
    out_path = config.RESULTS_DIR / "ablation.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Ablation results saved to {out_path}")


if __name__ == "__main__":
    main()
