import sys
from colabdesign import mk_afdesign_model, clear_mem
OUT = "/workspace/stage_gpu_complex.pdb"
clear_mem()
print("[gpu] building binder afdesign model (data_dir=/app, multimer)...", flush=True)
m = mk_afdesign_model(protocol="binder", debug=False, data_dir="/app",
                      use_multimer=True, num_recycles=1, best_metric="loss")
m.prep_inputs(pdb_filename="/app/example/PDL1.pdb", chain="A",
              binder_len=70, hotspot="56", seed=0)
print("[gpu] running 3 logit iterations (smoke)...", flush=True)
m.design_logits(iters=3, num_models=1, save_best=True)
m.save_pdb(OUT)
print(f"[gpu] WROTE {OUT}", flush=True)
chains = set()
with open(OUT) as fh:
    for line in fh:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            chains.add(line[21])
print(f"[gpu] chains in output PDB: {sorted(chains)}", flush=True)
sys.exit(0 if {"A", "B"} <= chains else 1)
