#!/usr/bin/env python
"""rosetta_service.py — HTTP service wrapping BindCraft's PyRosetta functions.

Runs in the bindcraft-rosetta image (PyRosetta + DAlphaBall + dssp + biopython, no jax).
The GPU worker process calls these endpoints via functions/rosetta_client.py. PDB files
are read/written on the shared /workspace volume (mounted at the same path here), so only
file paths travel over the wire.

Endpoints (POST JSON, returns {"ok": bool, "result": ..., "error": str}):
    /pr_relax          {pdb_file, relaxed_pdb_path}                         -> null
    /score_interface   {pdb_file, binder_chain}                            -> {scores, interface_AA, interface_residues}
    /unaligned_rmsd    {reference_pdb, align_pdb, reference_chain_id, align_chain_id} -> {rmsd}
    /align_pdbs        {reference_pdb, align_pdb, reference_chain_id, align_chain_id} -> null
    GET /health        -> 200 "ok"

PyRosetta is initialised once at startup with the same flags as bindcraft.py:60.
Calls are serialised with a lock (PyRosetta is not thread-safe); scale by running
multiple replicas of this service behind the compose service DNS.
"""
import os
import json
import threading
# NOTE: a single-threaded HTTPServer is used on purpose. PyRosetta movers (e.g. FastRelax)
# segfault when executed off the main thread in the current PyRosetta build, so a
# ThreadingHTTPServer (one worker thread per request) crashes the whole service on the
# first /pr_relax call. Handling requests synchronously on the main thread avoids this,
# and costs nothing: PyRosetta calls are already serialised by _LOCK below, and throughput
# is scaled by running multiple service replicas (see ROSETTA_REPLICAS / compose DNS).
from http.server import BaseHTTPRequestHandler, HTTPServer

import pyrosetta as pr

# Initialise PyRosetta once, mirroring bindcraft.py:60.
DALPHABALL_PATH = os.environ.get("DALPHABALL_PATH", "/app/functions/DAlphaBall.gcc")
pr.init(
    f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
    f'-holes:dalphaball {DALPHABALL_PATH} -corrections::beta_nov16 true -relax:default_repeats 1'
)

# Import the real implementations (these run here, in the PyRosetta env).
from functions.pyrosetta_utils import pr_relax, score_interface, unaligned_rmsd, align_pdbs

_LOCK = threading.Lock()
PORT = int(os.environ.get("ROSETTA_PORT", "8000"))


def _handle(endpoint, p):
    if endpoint == "pr_relax":
        pr_relax(p["pdb_file"], p["relaxed_pdb_path"])
        return None
    if endpoint == "score_interface":
        scores, interface_AA, interface_residues = score_interface(p["pdb_file"], p.get("binder_chain", "B"))
        return {"scores": scores, "interface_AA": interface_AA, "interface_residues": interface_residues}
    if endpoint == "unaligned_rmsd":
        rmsd = unaligned_rmsd(p["reference_pdb"], p["align_pdb"], p["reference_chain_id"], p["align_chain_id"])
        return {"rmsd": rmsd}
    if endpoint == "align_pdbs":
        align_pdbs(p["reference_pdb"], p["align_pdb"], p["reference_chain_id"], p["align_chain_id"])
        return None
    raise ValueError(f"unknown endpoint: {endpoint}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, {"ok": True, "result": "ok"})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        endpoint = self.path.strip("/")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            with _LOCK:  # serialise PyRosetta calls
                result = _handle(endpoint, payload)
            self._send(200, {"ok": True, "result": result})
        except Exception as e:  # return the error to the client instead of dropping the connection
            import traceback
            traceback.print_exc()
            self._send(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def _send(self, code, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    print(f"[rosetta] PyRosetta initialised (dalphaball={DALPHABALL_PATH}); listening on :{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
