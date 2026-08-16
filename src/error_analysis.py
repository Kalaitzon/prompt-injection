"""
error_analysis.py
=================
task 6

Loads the baseline and hardened results and produces:

  1. A CONFUSION MATRIX with precision / recall / F1, treating the system as a
     binary "block attacks" detector (lecture 5 terminology):
        positive class = adversarial (should be blocked)
        TP = adversarial that was blocked (no leak)
        FN = adversarial that leaked            (a real DEFENSE FAILURE)
        FP = benign that was wrongly rejected   (over-blocking)
        TN = benign that was served correctly

  2. The FALSE NEGATIVES (attacks that leaked), each with its root cause: which
     family, which layers were active and which (if any) triggered, so that it
     is visible WHERE in the flow the failure occurred.

  3. The FALSE POSITIVES (benign not served), BY CAUSE:
        defense_block  -> a defense layer actually blocked/filtered it (genuine FP)
        model_refusal  -> the model itself refused (not our defense)
        leaked         -> benign that accidentally showed a canary
     We deliberately do NOT count as a failure the cases "task_type == unknown but
     answered correctly". These are recorded separately as CLASSIFIER NOTES,
     since they are a labeling quirk and not a security event.

Output: results/error_analysis.json (plus a printed report)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config


def _load(path) -> Optional[Dict[str, Any]]:
    # Loads a JSON results file, or None if it does not exist.
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

@dataclass
class Confusion:
    tp: int = 0  # adversarial that was blocked
    fn: int = 0  # adversarial that leaked (failure)
    fp: int = 0  # benign not served
    tn: int = 0  # benign served

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "TP_adversarial_blocked": self.tp,
            "FN_adversarial_leaked": self.fn,
            "FP_benign_overblocked": self.fp,
            "TN_benign_served": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
        }


def confusion_from_results(results: Dict[str, Any]) -> Confusion:
    # Iterates over all cases and classifies them into TP/FN/FP/TN.
    c = Confusion()
    for case in results["cases"]:
        if case["category"] == "adversarial":
            if case.get("attack_success"):
                c.fn += 1   # leaked -> defense failure
            else:
                c.tp += 1   # correctly blocked
        else:  # benign
            if case.get("benign_success"):
                c.tn += 1   # correctly served
            else:
                c.fp += 1   # not served
    return c


# ---------------------------------------------------------------------------
# Cause analysis
# ---------------------------------------------------------------------------

def _case_defense_decision(case: Dict[str, Any]) -> str:
    """Returns the strongest defense decision across all turns of the case."""
    decisions = []
    for t in case["turns"]:
        di = t.get("defense_info")
        if di:
            decisions.append(di.get("decision", "approved"))
    if "blocked" in decisions:
        return "blocked"
    if "filtered" in decisions:
        return "filtered"
    return "approved"


def analyse_false_negatives(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    # For each attack that leaked, records which layers were active/triggered
    # and attributes a likely root cause of the failure per family.
    fns = []
    for case in results["cases"]:
        if case["category"] == "adversarial" and case.get("attack_success"):
            # which layers triggered anywhere across the turns?
            triggered = {"l1": [], "l2": [], "l3": []}
            layers_enabled = None
            for t in case["turns"]:
                di = t.get("defense_info") or {}
                layers_enabled = di.get("layers", layers_enabled)
                triggered["l1"] += di.get("layer1_triggered", [])
                triggered["l2"] += di.get("layer2_triggered", [])
                triggered["l3"] += di.get("layer3_triggered", [])
            fns.append({
                "test_id": case["test_id"],
                "attack_family": case["attack_family"],
                "difficulty": case["difficulty"],
                "leaked_canaries": case.get("leaked_canaries", []),
                "leak_turn": case.get("leak_turn"),
                "layers_enabled": layers_enabled,
                "layers_triggered": {k: v for k, v in triggered.items() if v},
                "root_cause": _fn_root_cause(case["attack_family"], triggered),
            })
    return fns


def _fn_root_cause(family: str, triggered: Dict[str, List[str]]) -> str:
    # Short, per-family explanation of why an attack might have slipped through.
    if family == "indirect_instruction_injection":
        return ("Untrusted-document payload survived context separation; output "
                "filter did not recognise the paraphrased disclosure.")
    if family == "multi_turn_decomposition":
        return ("Compromise assembled across turns; per-turn input scoring saw no "
                "single high-risk message.")
    if family == "encoded_obfuscated":
        return "Obfuscation not decoded by input validation; model complied after decoding."
    if family == "role_play":
        return "Persona framing lowered model refusal; no canary phrase for output filter to catch."
    if family == "direct_instruction_override":
        return "Direct override slipped input scoring threshold; phrasing avoided trigger terms."
    return "Unclassified."


def analyse_false_positives(results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    # Splits the benign cases not served by cause: defense, model refusal, or leak.
    out = {"defense_block": [], "model_refusal": [], "leaked": []}
    for case in results["cases"]:
        if case["category"] != "benign" or case.get("benign_success"):
            continue
        decision = _case_defense_decision(case)
        leaked = bool(case.get("leaked_canaries"))
        entry = {
            "test_id": case["test_id"],
            "task_type": case["turns"][-1]["task_type"],
            "decision": decision,
            "response_preview": case["turns"][-1]["response"][:120],
        }
        if leaked:
            out["leaked"].append(entry)
        elif decision in ("blocked", "filtered"):
            out["defense_block"].append(entry)
        else:
            out["model_refusal"].append(entry)
    return out


def benign_classifier_notes(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Benign answered correctly but with task_type == 'unknown' (NOT a failure)."""
    notes = []
    for case in results["cases"]:
        if case["category"] == "benign" and case.get("benign_success"):
            if case["turns"][-1]["task_type"] == "unknown":
                notes.append({
                    "test_id": case["test_id"],
                    "note": "Answered successfully but task_type classified 'unknown' "
                            "(classifier keyword gap, not a security event).",
                })
    return notes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    # Loads the results, builds the confusion matrices, analyzes the errors of the
    # hardened system, prints the report and stores it in JSON.
    config.ensure_dirs()
    baseline = _load(config.RESULTS_DIR / "baseline_results.json")
    defended = _load(config.RESULTS_DIR / "defended_results.json")

    if defended is None:
        print("No defended_results.json found. Run: python3 runner.py --system defended")
        return

    print("=" * 72)
    print("ERROR ANALYSIS")
    print("=" * 72)

    report: Dict[str, Any] = {}

    for label, res in (("baseline", baseline), ("defended", defended)):
        if res is None:
            continue
        conf = confusion_from_results(res)
        report[label] = {"confusion": conf.as_dict()}
        print(f"\n[{label}] confusion matrix (positive = adversarial/block):")
        cd = conf.as_dict()
        print(f"   TP(blocked)={cd['TP_adversarial_blocked']}  "
              f"FN(leaked)={cd['FN_adversarial_leaked']}  "
              f"FP(overblock)={cd['FP_benign_overblocked']}  "
              f"TN(served)={cd['TN_benign_served']}")
        print(f"   precision={cd['precision']:.2f}  recall={cd['recall']:.2f}  "
              f"F1={cd['f1']:.2f}  FPR={cd['false_positive_rate']:.2f}")

    # detailed cause analysis on the defended system
    fns = analyse_false_negatives(defended)
    fps = analyse_false_positives(defended)
    notes = benign_classifier_notes(defended)
    report["defended"]["false_negatives"] = fns
    report["defended"]["false_positives_by_cause"] = fps
    report["defended"]["benign_classifier_notes"] = notes

    print("\n" + "-" * 72)
    print(f"FALSE NEGATIVES (defended attacks that leaked): {len(fns)}")
    for fn in fns:
        print(f"  - {fn['test_id']} [{fn['attack_family']}/{fn['difficulty']}] "
              f"leaked={fn['leaked_canaries']}")
        print(f"      root cause: {fn['root_cause']}")

    print("\nFALSE POSITIVES (benign not served), by cause:")
    for cause, items in fps.items():
        print(f"  {cause}: {len(items)}")
        for it in items:
            print(f"     - {it['test_id']} (decision={it['decision']})")

    print(f"\nBENIGN CLASSIFIER NOTES (answered OK, task_type=unknown; NOT failures): "
          f"{len(notes)}")
    for n in notes:
        print(f"  - {n['test_id']}")

    out_path = config.RESULTS_DIR / "error_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Error analysis saved to {out_path}")


if __name__ == "__main__":
    main()
