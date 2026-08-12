# Ioannis Kalaitzidis, MTE25012

"""
ablation.py
===========
task 5

Ο θωρακισμένος βοηθός έχει τρία ανεξάρτητα επίπεδα άμυνας (L1 έλεγχος εισόδου,
L2 διαχωρισμός πλαισίου, L3 φιλτράρισμα εξόδου). Το αρχείο αυτό δοκιμάζει και τους
8 συνδυασμούς ενεργοποίησης/απενεργοποίησης των τριών επιπέδων (2^3 = 8) και για
καθέναν αναφέρει την επιτυχία/μπλοκάρισμα επιθέσεων, την καλόπιστη επιτυχία, τις
εσφαλμένες αρνήσεις, και την επιτυχία επίθεσης ανά οικογένεια.

Κάθε συνδυασμός εκτελείται N_RUNS φορές και το αποτέλεσμα δίνεται ως μέσος όρος
+/- τυπική απόκλιση, ώστε να φαίνεται και η διασπορά και όχι μόνο ένα νούμερο.
Υπό temperature=0 οι επαναλήψεις βγαίνουν ταυτόσημες, οπότε η τυπική απόκλιση
είναι 0, κάτι που το δηλώνουμε ρητά ως ένδειξη αναπαραγωγιμότητας.

Εκτός από τους 8 συνδυασμούς, υπολογίζεται και η οριακή συνεισφορά κάθε επιπέδου,
δηλαδή η μέση επίδραση που έχει η ενεργοποίησή του όταν τα άλλα δύο μένουν σταθερά.
Αυτό δείχνει πόσο αξίζει κάθε επίπεδο ξεχωριστά, χωρίς να βασιζόμαστε σε μία μόνο
διαμόρφωση.

Είσοδος: το corpus του benchmark (από το benchmark_generator.py)
Έξοδος:  results/ablation.json και πίνακες στην κονσόλα
"""

from __future__ import annotations

import json
import statistics
from itertools import product
from typing import Any, Dict, List, Tuple

import config
from runner import run_case, compute_summary


# Πλήθος επαναλήψεων ανά διαμόρφωση. Προεπιλογή 3, αλλά ρυθμίζεται μέσω της μεταβλητής
# περιβάλλοντος BANK_ABLATION_RUNS (π.χ. =1 για γρήγορη ντετερμινιστική εκτέλεση, αφού
# υπό temperature=0 κάθε επανάληψη είναι ούτως ή άλλως ταυτόσημη).
N_RUNS = int(__import__("os").environ.get("BANK_ABLATION_RUNS", "3"))

# Οι 8 συνδυασμοί επιπέδων, με κλειδί την τριάδα (L1, L2, L3) on/off, αντιστοιχισμένοι
# σε αναγνώσιμη ετικέτα που χρησιμοποιείται στους πίνακες και στο results/ablation.json.
CONFIG_NAMES = {
    (False, False, False): "None (baseline)",   # καμία άμυνα -> ισοδυναμεί με το baseline του Ζητ. 3
    (True,  False, False): "L1 only (input)",
    (False, True,  False): "L2 only (context)",
    (False, False, True):  "L3 only (output)",
    (True,  True,  False): "L1+L2",
    (True,  False, True):  "L1+L3",
    (False, True,  True):  "L2+L3",
    (True,  True,  True):  "L1+L2+L3 (full)",
}


def _load_cases() -> List[Dict[str, Any]]:
    # Φόρτωση του corpus και συνένωση καλόπιστων + ανταγωνιστικών περιπτώσεων σε μία
    # λίστα, ώστε κάθε διαμόρφωση να δοκιμάζεται στις ίδιες ακριβώς 60 περιπτώσεις.
    corpus = json.loads(config.CORPUS_JSON.read_text(encoding="utf-8"))
    return corpus["benign_prompts"] + corpus["adversarial_prompts"]


def _evaluate_once(l1: bool, l2: bool, l3: bool, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Εκτελεί το πλήρες benchmark μία φορά για μια δεδομένη διαμόρφωση επιπέδων (σιωπηλά)."""
    # Κατασκευή θωρακισμένου βοηθού με ενεργά ακριβώς τα ζητούμενα επίπεδα. Αυτό είναι
    # που καθιστά εφικτό το ablation: κάθε επίπεδο ενεργοποιείται ανεξάρτητα.
    from assistant_defended import DefendedAssistant
    assistant = DefendedAssistant(enable_layer1=l1, enable_layer2=l2, enable_layer3=l3,
                                  system_label=f"abl_{int(l1)}{int(l2)}{int(l3)}")
    # Εκτέλεση κάθε περίπτωσης μέσω αυτού του βοηθού και σύνοψη στις τυπικές μετρικές.
    results = [run_case(assistant, c) for c in cases]
    return compute_summary(results)


def run_config(l1: bool, l2: bool, l3: bool, cases: List[Dict[str, Any]],
               n_runs: int = N_RUNS) -> Dict[str, Any]:
    # Εκτέλεση μιας διαμόρφωσης n_runs φορές και συγκέντρωση. Η επανάληψη είναι που
    # επιτρέπει να αναφέρουμε διασπορά (std) αντί για ένα μεμονωμένο νούμερο.
    name = CONFIG_NAMES[(l1, l2, l3)]
    runs = [_evaluate_once(l1, l2, l3, cases) for _ in range(n_runs)]

    def agg(key: str) -> Dict[str, float]:
        # Μέσος όρος και τυπική απόκλιση (πληθυσμού) μιας μετρικής στις n_runs επαναλήψεις.
        vals = [r[key] for r in runs]
        return {
            "mean": round(statistics.fmean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        }

    # Μέση επιτυχία επίθεσης ανά οικογένεια: δείχνει ποιον τύπο επίθεσης σταματά κάθε διαμόρφωση.
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
    Μέση οριακή επίδραση της ενεργοποίησης κάθε επιπέδου, με τα άλλα δύο σταθερά.
    Θετικό 'attack_reduction' σημαίνει όφελος ασφάλειας, θετικό 'refusal_increase'
    σημαίνει κόστος χρηστικότητας.

    Για κάθε επίπεδο κοιτάμε τα 4 πλαίσια που δίνουν οι on/off καταστάσεις των άλλων
    δύο επιπέδων, μετράμε πόσο αλλάζει η μετρική όταν το ανάβουμε σε κάθε πλαίσιο, και
    παίρνουμε τον μέσο όρο. Έτσι η τιμή δεν εξαρτάται από μία μόνο διαμόρφωση.
    """
    out: Dict[str, Any] = {}
    for layer_idx, layer_name in enumerate(["L1_input", "L2_context", "L3_output"]):
        attack_deltas: List[float] = []
        refusal_deltas: List[float] = []
        # Το 'others' απαριθμεί τις 4 on/off καταστάσεις των άλλων δύο επιπέδων.
        for others in product([False, True], repeat=2):
            # Κατασκευή των δύο κλειδιών διαμόρφωσης που διαφέρουν ΜΟΝΟ σε αυτό το επίπεδο:
            # ένα με αυτό OFF και ένα με αυτό ON, με τα άλλα δύο επίπεδα ταυτόσημα.
            off_key = list(others); off_key.insert(layer_idx, False); off_key = tuple(off_key)
            on_key = list(others); on_key.insert(layer_idx, True); on_key = tuple(on_key)
            off = by_config[off_key]; on = by_config[on_key]
            # Όφελος ασφάλειας: πόσο ΜΕΙΩΝΕΤΑΙ η επιτυχία επίθεσης όταν το επίπεδο είναι ON.
            attack_deltas.append(off["attack_success_rate"]["mean"]
                                 - on["attack_success_rate"]["mean"])
            # Κόστος χρηστικότητας: πόσο ΑΥΞΑΝΟΝΤΑΙ οι εσφαλμένες αρνήσεις όταν το επίπεδο είναι ON.
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
    # Διασφάλιση ότι υπάρχουν οι φάκελοι εξόδου, και παραγωγή του corpus αν λείπει.
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

    # Εκτέλεση και των 8 διαμορφώσεων, κρατώντας τα αποτελέσματα και με κλειδί (για την
    # ανάλυση οριακής συνεισφοράς) και σε σειρά (για τους πίνακες εκτύπωσης/αποθήκευσης).
    by_config: Dict[Tuple[bool, bool, bool], Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    for (l1, l2, l3) in keys:
        res = run_config(l1, l2, l3, cases)
        by_config[(l1, l2, l3)] = res
        ordered.append(res)
        print(f"  done: {res['name']:<22} "
              f"attack={res['attack_success_rate']['mean']:.0%} "
              f"benign={res['benign_success_rate']['mean']:.0%}")

    # Υπολογισμός της οριακής συνεισφοράς κάθε επιπέδου από τις 8 διαμορφώσεις.
    marg = marginal_contributions(by_config)

    # Εκτύπωση των τριών αναγνώσιμων πινάκων (συνολικός / ανά οικογένεια / οριακός).
    print_table(ordered)
    print_family_table(ordered)
    print_marginal(marg)

    # Αποθήκευση όλων στο results/ablation.json (τα σχήματα διαβάζουν από αυτό το αρχείο).
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
