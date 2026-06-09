"""rosetta_client.py — HTTP client shim for the BindCraft Rosetta service.

Active only when BINDCRAFT_MODE=gpu (see functions/__init__.py and
functions/colabdesign_utils.py). Exposes the SAME names that the GPU process uses
from pyrosetta_utils, but forwards each call to the Rosetta service container over
HTTP. PDB files are exchanged via the shared /workspace volume (paths only travel
over the wire, not file contents), so the service must mount the same volume at the
same path.

Exported symbols (must match the pyrosetta_utils surface used outside that module):
    pr_relax(pdb_file, relaxed_pdb_path)            -> None  (writes relaxed PDB)
    score_interface(pdb_file, binder_chain="B")     -> (scores_dict, interface_AA, interface_residues_str)
    unaligned_rmsd(ref, align, ref_chain, align_ch) -> float
    align_pdbs(ref, align, ref_chain, align_chain)  -> None  (rewrites align PDB in place)
    pr                                              -> no-op shim; pr.init(...) does nothing
                                                       (PyRosetta is initialised inside the service)

Dependency-free (stdlib only) so the GPU image needs nothing extra.
"""
import os
import json
import time
import urllib.request
import urllib.error

ROSETTA_URL = os.environ.get("ROSETTA_URL", "http://rosetta:8000").rstrip("/")
_TIMEOUT = float(os.environ.get("ROSETTA_TIMEOUT", "3600"))   # FastRelax can be slow
_RETRIES = int(os.environ.get("ROSETTA_RETRIES", "30"))       # tolerate service warm-up
_BACKOFF = float(os.environ.get("ROSETTA_BACKOFF", "5"))


def _call(endpoint, payload):
    """POST JSON payload to the service and return the parsed JSON response."""
    url = f"{ROSETTA_URL}/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if not body.get("ok", False):
                raise RuntimeError(f"Rosetta service error on {endpoint}: {body.get('error')}")
            return body.get("result")
        except urllib.error.URLError as e:
            # connection refused / service not up yet -> retry with backoff
            last_err = e
            time.sleep(_BACKOFF)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Rosetta service HTTP {e.code} on {endpoint}: {e.read().decode('utf-8', 'ignore')}")
    raise RuntimeError(f"Rosetta service unreachable at {url} after {_RETRIES} attempts: {last_err}")


def wait_until_ready():
    """Block until the service /health endpoint responds (used at worker startup)."""
    url = f"{ROSETTA_URL}/health"
    for _ in range(_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.URLError:
            time.sleep(_BACKOFF)
    return False


def pr_relax(pdb_file, relaxed_pdb_path):
    _call("pr_relax", {"pdb_file": pdb_file, "relaxed_pdb_path": relaxed_pdb_path})


def score_interface(pdb_file, binder_chain="B"):
    r = _call("score_interface", {"pdb_file": pdb_file, "binder_chain": binder_chain})
    return r["scores"], r["interface_AA"], r["interface_residues"]


def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    r = _call("unaligned_rmsd", {
        "reference_pdb": reference_pdb, "align_pdb": align_pdb,
        "reference_chain_id": reference_chain_id, "align_chain_id": align_chain_id,
    })
    return r["rmsd"]


def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    _call("align_pdbs", {
        "reference_pdb": reference_pdb, "align_pdb": align_pdb,
        "reference_chain_id": reference_chain_id, "align_chain_id": align_chain_id,
    })


class _PyRosettaShim:
    """Stand-in for `pyrosetta as pr` in remote mode. The real pr.init() runs in the
    service; here it is a no-op so bindcraft.py's `pr.init(...)` call is harmless."""
    def init(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        raise AttributeError(
            f"pyrosetta.{name} is not available in remote mode; this call must run "
            f"inside the Rosetta service, not the GPU process."
        )


pr = _PyRosettaShim()
