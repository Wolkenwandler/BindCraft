import os, csv
import pyrosetta as pr
pr.init('-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
        '-holes:dalphaball /app/functions/DAlphaBall.gcc -corrections::beta_nov16 true -relax:default_repeats 1')
from functions.pyrosetta_utils import pr_relax, score_interface
IN = "/workspace/stage_gpu_complex.pdb"
RELAXED = "/workspace/stage_relaxed.pdb"
DP = "/workspace/designs/PDL1_smoke"
os.makedirs(DP, exist_ok=True)
assert os.path.exists(IN), f"missing gpu output {IN}"
print(f"[rosetta] relaxing {IN} -> {RELAXED}", flush=True)
pr_relax(IN, RELAXED)
print("[rosetta] scoring interface (binder chain B)...", flush=True)
scores, interface_AA, interface_residues = score_interface(RELAXED, "B")
dG = scores.get("interface_dG"); nres = scores.get("interface_nres")
print(f"[rosetta] interface_dG={dG} interface_nres={nres}", flush=True)
csv_path = os.path.join(DP, "final_design_stats.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["Design", "interface_dG", "interface_nres"])
    w.writerow(["PDL1smoke_stage", dG, nres])
print(f"[rosetta] WROTE {csv_path}", flush=True)
