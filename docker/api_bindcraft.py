"""
BindCraft API Server

Wraps BindCraft's core protein binder design pipeline as HTTP endpoints.
Runs inside the GPU image (BINDCRAFT_MODE=gpu); PyRosetta calls are forwarded
to the Rosetta service container over HTTP.

References:
  - cardiorgan/api_cardiorgan.py for the FastAPI pattern
  - bindcraft.py for the design loop orchestration
"""

import os
import sys
import re
import json
import time
import shutil
import math
import gc
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure BINDCRAFT_MODE=gpu so that functions/__init__.py loads the right stack
os.environ.setdefault("BINDCRAFT_MODE", "gpu")

# Add the parent directory to sys.path so we can import `functions`
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from functions import *

# ==============================================================================
# FastAPI app
# ==============================================================================
app = FastAPI(
    title="BindCraft API",
    description="De novo protein binder design API — hallucination, MPNN redesign, AF2 validation",
    version="1.0.0",
)

# ==============================================================================
# Pydantic models
# ==============================================================================

class TargetSettings(BaseModel):
    """Target protein settings (same schema as settings_target/*.json)."""
    design_path: str = "/workspace/designs/default"
    binder_name: str = "binder"
    starting_pdb: str
    chains: str = "A"
    target_hotspot_residues: str = ""
    lengths: List[int] = [65, 150]
    number_of_final_designs: int = 1

class AdvancedSettings(BaseModel):
    """Advanced design protocol settings (same schema as settings_advanced/*.json)."""
    # Design algorithm
    design_algorithm: str = "4stage"
    use_multimer_design: bool = True

    # Iterations
    soft_iterations: int = 100
    temporary_iterations: int = 50
    hard_iterations: int = 0
    greedy_iterations: int = 25
    greedy_percentage: int = 10

    # Model params
    num_recycles_design: int = 3
    num_recycles_validation: int = 3
    sample_models: bool = False

    # Loss weights
    weights_plddt: float = 1.0
    weights_pae_intra: float = 0.0
    weights_pae_inter: float = 0.0
    weights_con_intra: float = 0.0
    weights_con_inter: float = 0.5

    # Contact settings
    intra_contact_number: int = 40
    intra_contact_distance: int = 8
    inter_contact_number: int = 50
    inter_contact_distance: int = 8

    # Additional losses
    use_rg_loss: bool = False
    weights_rg: float = 0.0
    use_i_ptm_loss: bool = False
    weights_iptm: float = 0.0
    use_termini_distance_loss: bool = False
    weights_termini_loss: float = 0.0
    random_helicity: bool = False
    weights_helicity: float = 0.0

    # Template settings
    rm_template_seq_design: bool = True
    rm_template_sc_design: bool = False
    rm_template_seq_predict: bool = True
    rm_template_sc_predict: bool = False

    # MPNN settings
    enable_mpnn: bool = True
    num_seqs: int = 30
    sampling_temp: float = 0.1
    backbone_noise: float = 0.0
    mpnn_fix_interface: bool = True
    max_mpnn_sequences: int = 5
    force_reject_AA: bool = False
    omit_AAs: Optional[str] = None

    # Prediction settings
    predict_initial_guess: bool = True
    predict_bigbang: bool = False

    # Beta optimisation
    optimise_beta: bool = True
    optimise_beta_recycles_design: int = 16
    optimise_beta_recycles_valid: int = 4
    optimise_beta_extra_soft: int = 100
    optimise_beta_extra_temp: int = 100

    # Paths (auto-filled)
    af_params_dir: str = ""
    dssp_path: str = ""
    dalphaball_path: str = ""
    mpnn_weights: str = ""
    model_path: str = "v_48_020"

    # Runtime controls
    max_trajectories: Any = False
    start_monitoring: int = 50
    enable_rejection_check: bool = False
    acceptance_rate: float = 0.05

    # Output flags
    save_mpnn_fasta: bool = True
    save_design_trajectory_plots: bool = True
    save_design_animations: bool = True
    save_trajectory_pickle: bool = False
    remove_unrelaxed_complex: bool = False
    remove_binder_monomer: bool = False
    remove_unrelaxed_trajectory: bool = False
    zip_animations: bool = False
    zip_plots: bool = False

class FiltersModel(BaseModel):
    """Pass/fail filter thresholds (same schema as settings_filters/*.json)."""
    model_config = {"extra": "allow"}

class RunDesignRequest(BaseModel):
    target_settings: TargetSettings
    advanced_settings: Optional[AdvancedSettings] = None
    filters: Optional[Dict[str, Any]] = None

class HallucinateRequest(BaseModel):
    binder_name: str = "binder"
    starting_pdb: str
    chains: str = "A"
    target_hotspot_residues: str = ""
    length: int = 100
    seed: Optional[int] = None
    design_path: str = "/workspace/designs/default"
    advanced_settings: Optional[Dict[str, Any]] = None

class PredictComplexRequest(BaseModel):
    binder_sequence: str
    target_pdb: str
    target_chain: str = "A"
    binder_length: int
    trajectory_pdb: str
    design_path: str = "/workspace/designs/default"
    design_name: str = "prediction"
    advanced_settings: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None

# ==============================================================================
# Global state — initialised once at startup
# ==============================================================================
_default_advanced: Dict[str, Any] = {}
_default_filters: Dict[str, Any] = {}
_design_models: List[int] = []
_prediction_models: List[int] = []
_multimer_validation: bool = False


def _load_defaults():
    """Load default advanced settings and filters from the bundled JSON files."""
    global _default_advanced, _default_filters
    bindcraft_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adv_path = os.environ.get(
        "DEFAULT_ADVANCED",
        os.path.join(bindcraft_dir, "settings_advanced", "default_4stage_multimer.json"),
    )
    fil_path = os.environ.get(
        "DEFAULT_FILTERS",
        os.path.join(bindcraft_dir, "settings_filters", "default_filters.json"),
    )
    with open(adv_path) as f:
        _default_advanced = json.load(f)
    with open(fil_path) as f:
        _default_filters = json.load(f)


def _init_models():
    """Initialize AF2 models and PyRosetta."""
    global _design_models, _prediction_models, _multimer_validation
    _design_models, _prediction_models, _multimer_validation = load_af2_models(
        _default_advanced.get("use_multimer_design", True)
    )
    # pr.init is a no-op in gpu mode
    dalphaball = os.environ.get("DALPHABALL_PATH", "/app/functions/DAlphaBall.gcc")
    pr.init(
        f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
        f'-holes:dalphaball {dalphaball} -corrections::beta_nov16 true '
        f'-relax:default_repeats 1'
    )


def _check_advanced(adv: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in paths for advanced settings."""
    bindcraft_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return perform_advanced_settings_check(adv, bindcraft_dir)


# ==============================================================================
# Helper: merge user settings with defaults
# ==============================================================================

def _merge_target(t: TargetSettings) -> Dict:
    return {
        "design_path": t.design_path,
        "binder_name": t.binder_name,
        "starting_pdb": t.starting_pdb,
        "chains": t.chains,
        "target_hotspot_residues": t.target_hotspot_residues,
        "lengths": t.lengths,
        "number_of_final_designs": t.number_of_final_designs,
    }

def _merge_advanced(a: Optional[AdvancedSettings]) -> Dict:
    if a is None:
        return dict(_default_advanced)
    merged = dict(_default_advanced)
    merged.update({k: v for k, v in a.model_dump().items() if v is not None})
    return merged

def _merge_filters(f: Optional[Dict]) -> Dict:
    if f is None:
        return dict(_default_filters)
    return {**dict(_default_filters), **f}


# ==============================================================================
# Core pipeline functions
# ==============================================================================

def _run_single_trajectory(
    design_name: str,
    target_settings: Dict[str, Any],
    advanced_settings: Dict[str, Any],
    filters: Dict[str, Any],
    design_paths: Dict[str, str],
    trajectory_csv: str,
    mpnn_csv: str,
    final_csv: str,
    failure_csv: str,
    trajectory_labels: List[str],
    design_labels: List[str],
    final_labels: List[str],
    length: int,
    seed: int,
    helicity_value: float,
    settings_file: str,
    filters_file: str,
    advanced_file: str,
) -> Dict[str, Any]:
    """
    Run a single design trajectory: hallucinate → relax → score → MPNN → validate → filter.
    Returns a summary dict with design results.
    """
    trajectory_start_time = time.time()
    result = {
        "design_name": design_name,
        "status": "unknown",
        "accepted_designs": 0,
        "accepted_names": [],
    }

    # ---- Hallucinate ----
    trajectory = binder_hallucination(
        design_name, target_settings["starting_pdb"], target_settings["chains"],
        target_settings["target_hotspot_residues"], length, seed, helicity_value,
        _design_models, advanced_settings, design_paths, failure_csv,
    )
    trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
    trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
    trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}

    trajectory_time = time.time() - trajectory_start_time
    print(f"Trajectory {design_name} took: {trajectory_time:.1f}s")

    # If trajectory was terminated (clashes / low confidence), return early
    if trajectory.aux["log"]["terminate"] != "":
        result["status"] = trajectory.aux["log"]["terminate"]
        result["reason"] = f"Trajectory terminated: {trajectory.aux['log']['terminate']}"
        return result

    # ---- Relax + Score ----
    trajectory_relaxed = os.path.join(design_paths["Trajectory/Relaxed"], design_name + ".pdb")
    pr_relax(trajectory_pdb, trajectory_relaxed)

    binder_chain = "B"
    num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
    num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

    (trajectory_alpha, trajectory_beta, trajectory_loops,
     trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface,
     trajectory_i_plddt, trajectory_ss_plddt) = calc_ss_percentage(
        trajectory_pdb, advanced_settings, binder_chain)

    trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(
        trajectory_relaxed, binder_chain)

    trajectory_sequence = trajectory.get_seq(get_best=True)[0]
    traj_seq_notes = validate_design_sequence(trajectory_sequence, num_clashes_relaxed, advanced_settings)
    trajectory_target_rmsd = target_pdb_rmsd(trajectory_pdb, target_settings["starting_pdb"], target_settings["chains"])

    # Save trajectory stats
    trajectory_data = [
        design_name, advanced_settings["design_algorithm"], length, seed, helicity_value,
        target_settings["target_hotspot_residues"], trajectory_sequence, trajectory_interface_residues,
        trajectory_metrics['plddt'], trajectory_metrics['ptm'], trajectory_metrics['i_ptm'],
        trajectory_metrics['pae'], trajectory_metrics['i_pae'],
        trajectory_i_plddt, trajectory_ss_plddt, num_clashes_trajectory, num_clashes_relaxed,
        trajectory_interface_scores['binder_score'],
        trajectory_interface_scores['surface_hydrophobicity'], trajectory_interface_scores['interface_sc'],
        trajectory_interface_scores['interface_packstat'],
        trajectory_interface_scores['interface_dG'], trajectory_interface_scores['interface_dSASA'],
        trajectory_interface_scores['interface_dG_SASA_ratio'],
        trajectory_interface_scores['interface_fraction'], trajectory_interface_scores['interface_hydrophobicity'],
        trajectory_interface_scores['interface_nres'], trajectory_interface_scores['interface_interface_hbonds'],
        trajectory_interface_scores['interface_hbond_percentage'],
        trajectory_interface_scores['interface_delta_unsat_hbonds'],
        trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
        trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface,
        trajectory_alpha, trajectory_beta, trajectory_loops, trajectory_interface_AA,
        trajectory_target_rmsd,
        f"{trajectory_time:.1f}s", traj_seq_notes, settings_file, filters_file, advanced_file,
    ]
    insert_data(trajectory_csv, trajectory_data)

    if not trajectory_interface_residues:
        result["status"] = "no_interface"
        result["reason"] = "No interface residues found, skipping MPNN"
        return result

    if not advanced_settings.get("enable_mpnn", True):
        result["status"] = "success_no_mpnn"
        return result

    # ---- MPNN redesign + validation ----
    mpnn_n = 1
    accepted_mpnn = 0
    mpnn_start_time = time.time()

    mpnn_trajectories = mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings)
    existing_mpnn_sequences = set(
        pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values
        if os.path.exists(mpnn_csv) and os.path.getsize(mpnn_csv) > 0 else []
    )

    restricted_AAs = (
        set(aa.strip().upper() for aa in advanced_settings.get("omit_AAs", "").split(','))
        if advanced_settings.get("omit_AAs") and advanced_settings.get("force_reject_AA") else set()
    )

    mpnn_sequences = sorted({
        mpnn_trajectories['seq'][n][-length:]: {
            'seq': mpnn_trajectories['seq'][n][-length:],
            'score': mpnn_trajectories['score'][n],
            'seqid': mpnn_trajectories['seqid'][n],
        }
        for n in range(min(advanced_settings.get("num_seqs", 30), len(mpnn_trajectories.get('seq', []))))
        if (not restricted_AAs or
            not any(aa in mpnn_trajectories['seq'][n][-length:].upper() for aa in restricted_AAs))
        and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
    }.values(), key=lambda x: x['score'])

    del existing_mpnn_sequences

    if not mpnn_sequences:
        result["status"] = "no_mpnn_sequences"
        result["reason"] = "All MPNN sequences were duplicates or filtered out"
        return result

    # Beta optimisation
    if (advanced_settings.get("optimise_beta", False) and
            float(trajectory_beta) > 15):
        advanced_settings["num_recycles_validation"] = advanced_settings.get(
            "optimise_beta_recycles_valid", 4)

    # Compile prediction models
    clear_mem()
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings.get("num_recycles_validation", 3),
        data_dir=advanced_settings.get("af_params_dir", "/app"),
        use_multimer=_multimer_validation,
        use_initial_guess=advanced_settings.get("predict_initial_guess", True),
        use_initial_atom_pos=advanced_settings.get("predict_bigbang", False),
    )
    if advanced_settings.get("predict_initial_guess") or advanced_settings.get("predict_bigbang"):
        complex_prediction_model.prep_inputs(
            pdb_filename=trajectory_pdb, chain='A', binder_chain='B',
            binder_len=length, use_binder_template=True,
            rm_target_seq=advanced_settings.get("rm_template_seq_predict", True),
            rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
            rm_template_ic=True,
        )
    else:
        complex_prediction_model.prep_inputs(
            pdb_filename=target_settings["starting_pdb"],
            chain=target_settings["chains"],
            binder_len=length,
            rm_target_seq=advanced_settings.get("rm_template_seq_predict", True),
            rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
        )

    binder_prediction_model = mk_afdesign_model(
        protocol="hallucination", use_templates=False,
        initial_guess=False, use_initial_atom_pos=False,
        num_recycles=advanced_settings.get("num_recycles_validation", 3),
        data_dir=advanced_settings.get("af_params_dir", "/app"),
        use_multimer=_multimer_validation,
    )
    binder_prediction_model.prep_inputs(length=length)

    for mpnn_sequence in mpnn_sequences:
        mpnn_time = time.time()
        mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
        mpnn_score = round(mpnn_sequence['score'], 2)
        mpnn_seqid = round(mpnn_sequence['seqid'], 2)

        if advanced_settings.get("save_mpnn_fasta", True):
            save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)

        # Predict complex
        mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(
            complex_prediction_model, mpnn_sequence['seq'], mpnn_design_name,
            target_settings["starting_pdb"], target_settings["chains"],
            length, trajectory_pdb, _prediction_models, advanced_settings,
            filters, design_paths, failure_csv,
        )

        if not pass_af2_filters:
            mpnn_n += 1
            continue

        # Per-model scoring
        for model_num in _prediction_models:
            mpnn_design_pdb = os.path.join(
                design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            mpnn_design_relaxed = os.path.join(
                design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")

            if os.path.exists(mpnn_design_pdb):
                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)
                mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(
                    mpnn_design_relaxed, binder_chain)
                (mpnn_alpha, mpnn_beta, mpnn_loops,
                 mpnn_alpha_interface, mpnn_beta_interface, mpnn_loops_interface,
                 mpnn_i_plddt, mpnn_ss_plddt) = calc_ss_percentage(
                    mpnn_design_pdb, advanced_settings, binder_chain)
                rmsd_site = unaligned_rmsd(trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain)
                target_rmsd = target_pdb_rmsd(mpnn_design_pdb, target_settings["starting_pdb"], target_settings["chains"])

                mpnn_complex_statistics[model_num+1].update({
                    'i_pLDDT': mpnn_i_plddt, 'ss_pLDDT': mpnn_ss_plddt,
                    'Unrelaxed_Clashes': num_clashes_mpnn,
                    'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                    'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                    'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                    'PackStat': mpnn_interface_scores['interface_packstat'],
                    'dG': mpnn_interface_scores['interface_dG'],
                    'dSASA': mpnn_interface_scores['interface_dSASA'],
                    'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                    'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                    'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                    'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                    'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                    'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': mpnn_interface_AA,
                    'Interface_Helix%': mpnn_alpha_interface,
                    'Interface_BetaSheet%': mpnn_beta_interface,
                    'Interface_Loop%': mpnn_loops_interface,
                    'Binder_Helix%': mpnn_alpha,
                    'Binder_BetaSheet%': mpnn_beta,
                    'Binder_Loop%': mpnn_loops,
                    'Hotspot_RMSD': rmsd_site,
                    'Target_RMSD': target_rmsd,
                })

                if advanced_settings.get("remove_unrelaxed_complex"):
                    os.remove(mpnn_design_pdb)

        mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)

        # Predict binder alone
        binder_statistics = predict_binder_alone(
            binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name,
            length, trajectory_pdb, binder_chain, _prediction_models,
            advanced_settings, design_paths,
        )

        for model_num in _prediction_models:
            mpnn_binder_pdb = os.path.join(
                design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            rmsd_binder = None
            if os.path.exists(mpnn_binder_pdb):
                try:
                    rmsd_binder = unaligned_rmsd(trajectory_pdb, mpnn_binder_pdb, binder_chain, "A")
                except Exception as e:
                    print(f"Warning: binder RMSD failed for {mpnn_design_name}_model{model_num+1}: {e}")
            binder_statistics[model_num+1].update({'Binder_RMSD': rmsd_binder})
            if advanced_settings.get("remove_binder_monomer") and os.path.exists(mpnn_binder_pdb):
                os.remove(mpnn_binder_pdb)

        binder_averages = calculate_averages(binder_statistics)
        seq_notes = validate_design_sequence(mpnn_sequence['seq'], mpnn_complex_averages.get('Relaxed_Clashes'), advanced_settings)

        mpnn_end_time = time.time() - mpnn_time

        # Build mpnn_data row
        model_numbers = range(1, 6)
        statistics_labels = [
            'pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT',
            'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score',
            'Surface_Hydrophobicity', 'ShapeComplementarity', 'PackStat',
            'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity',
            'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage',
            'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%',
            'Binder_Helix%', 'Binder_BetaSheet%', 'Binder_Loop%',
            'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD',
        ]

        mpnn_data = [
            mpnn_design_name, advanced_settings["design_algorithm"], length, seed,
            helicity_value, target_settings["target_hotspot_residues"],
            mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid,
        ]

        for label in statistics_labels:
            mpnn_data.append(mpnn_complex_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))

        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
            mpnn_data.append(binder_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

        mpnn_data.extend([f"{mpnn_end_time:.1f}s", seq_notes, settings_file, filters_file, advanced_file])
        insert_data(mpnn_csv, mpnn_data)

        # Find best model
        plddt_values = {i: mpnn_data[i] for i in range(11, 16) if mpnn_data[i] is not None}
        if plddt_values:
            best_model_number = int(max(plddt_values, key=plddt_values.get)) - 10
            best_model_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb")

            filter_conditions = check_filters(mpnn_data, design_labels, filters)
            if filter_conditions is True:
                print(f"{mpnn_design_name} passed all filters")
                accepted_mpnn += 1
                shutil.copy(best_model_pdb, design_paths["Accepted"])
                final_data = [''] + mpnn_data
                insert_data(final_csv, final_data)
                result["accepted_names"].append(mpnn_design_name)

                if advanced_settings.get("save_design_animations"):
                    accepted_animation = os.path.join(design_paths["Accepted/Animation"], f"{design_name}.html")
                    src_animation = os.path.join(design_paths["Trajectory/Animation"], f"{design_name}.html")
                    if not os.path.exists(accepted_animation) and os.path.exists(src_animation):
                        shutil.copy(src_animation, accepted_animation)
            else:
                print(f"Unmet filter conditions for {mpnn_design_name}")
                if os.path.exists(best_model_pdb):
                    shutil.copy(best_model_pdb, design_paths["Rejected"])

        mpnn_n += 1
        if accepted_mpnn >= advanced_settings.get("max_mpnn_sequences", 5):
            break

    if advanced_settings.get("remove_unrelaxed_trajectory"):
        if os.path.exists(trajectory_pdb):
            os.remove(trajectory_pdb)

    total_design_time = time.time() - mpnn_start_time
    result["status"] = "success"
    result["accepted_designs"] = accepted_mpnn
    result["mpnn_total"] = mpnn_n - 1
    result["trajectory_time"] = round(trajectory_time, 1)
    result["design_time"] = round(total_design_time, 1)
    result["trajectory_pdb"] = trajectory_pdb
    return result


# ==============================================================================
# Startup
# ==============================================================================

@app.on_event("startup")
async def startup():
    """Initialize AF2 models and PyRosetta on startup."""
    print("[api] Loading default settings...")
    _load_defaults()
    print("[api] Checking GPU...")
    check_jax_gpu()
    print("[api] Initializing AF2 models + PyRosetta...")
    _init_models()
    print("[api] BindCraft API server ready.")


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check — returns GPU availability and service status."""
    import jax
    devices = jax.devices()
    gpu_devices = [str(d) for d in devices if d.platform == 'gpu']
    return {
        "status": "healthy",
        "service": "BindCraft API",
        "gpu_available": len(gpu_devices) > 0,
        "gpu_devices": gpu_devices,
        "bindcraft_mode": os.environ.get("BINDCRAFT_MODE", "gpu"),
    }


@app.post("/api/run_design")
async def api_run_design(request: RunDesignRequest):
    """
    Run the full BindCraft binder design pipeline.

    Accepts target settings, optional advanced settings and filters.
    Returns a summary of accepted designs and output paths.
    """
    try:
        target_settings = _merge_target(request.target_settings)
        advanced_settings = _check_advanced(_merge_advanced(request.advanced_settings))
        filters = _merge_filters(request.filters)

        settings_file = "api_target"
        filters_file = "api_filters"
        advanced_file = "api_advanced"

        # Setup directories and CSVs
        design_paths = generate_directories(target_settings["design_path"])
        trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

        trajectory_csv = os.path.join(target_settings["design_path"], "trajectory_stats.csv")
        mpnn_csv = os.path.join(target_settings["design_path"], "mpnn_design_stats.csv")
        final_csv = os.path.join(target_settings["design_path"], "final_design_stats.csv")
        failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

        create_dataframe(trajectory_csv, trajectory_labels)
        create_dataframe(mpnn_csv, design_labels)
        create_dataframe(final_csv, final_labels)

        # Build filters JSON and write it to a temp file for generate_filter_pass_csv
        filters_json_path = os.path.join(target_settings["design_path"], "_api_filters.json")
        with open(filters_json_path, 'w') as f:
            json.dump(filters, f)
        generate_filter_pass_csv(failure_csv, filters_json_path)

        script_start_time = time.time()
        trajectory_n = 1
        total_accepted = 0
        accepted_designs_list = []
        trajectory_results = []

        # Main design loop
        while True:
            # Check if we have enough final designs
            accepted_binders = [
                f for f in os.listdir(design_paths["Accepted"]) if f.endswith('.pdb')
            ]
            if len(accepted_binders) >= target_settings["number_of_final_designs"]:
                print(f"Target number of designs ({target_settings['number_of_final_designs']}) reached!")
                break

            # Check max trajectories
            if advanced_settings.get("max_trajectories"):
                n_trajectories = len([
                    f for f in os.listdir(design_paths["Trajectory/Relaxed"])
                    if f.endswith('.pdb')
                ])
                if n_trajectories >= advanced_settings["max_trajectories"]:
                    print("Max trajectories reached!")
                    break

            # Sample random params
            seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
            samples = np.arange(
                min(target_settings["lengths"]),
                max(target_settings["lengths"]) + 1,
            )
            length = int(np.random.choice(samples))
            helicity_value = load_helicity(advanced_settings)

            design_name = f"{target_settings['binder_name']}_l{length}_s{seed}"

            # Check if trajectory already exists
            trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
            trajectory_exists = any(
                os.path.exists(os.path.join(design_paths[d], design_name + ".pdb"))
                for d in trajectory_dirs
            )

            if trajectory_exists:
                trajectory_n += 1
                continue

            print(f"\n{'='*60}\nStarting trajectory {trajectory_n}: {design_name}\n{'='*60}")

            try:
                traj_result = _run_single_trajectory(
                    design_name, target_settings, advanced_settings, filters,
                    design_paths, trajectory_csv, mpnn_csv, final_csv, failure_csv,
                    trajectory_labels, design_labels, final_labels,
                    length, seed, helicity_value,
                    settings_file, filters_file, advanced_file,
                )
                trajectory_results.append(traj_result)
                total_accepted += traj_result.get("accepted_designs", 0)
                accepted_designs_list.extend(traj_result.get("accepted_names", []))
            except Exception as e:
                print(f"Trajectory {design_name} failed with error: {e}")
                traceback.print_exc()
                trajectory_results.append({
                    "design_name": design_name,
                    "status": "error",
                    "error": str(e),
                })

            trajectory_n += 1
            gc.collect()

        elapsed = time.time() - script_start_time
        elapsed_text = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"

        return {
            "status": "success",
            "msg": f"设计完成：{total_accepted} 个binder通过筛选",
            "design_path": target_settings["design_path"],
            "accepted_designs": total_accepted,
            "accepted_names": accepted_designs_list,
            "trajectories_run": trajectory_n - 1,
            "elapsed": elapsed_text,
            "elapsed_seconds": round(elapsed, 1),
            "final_csv": final_csv,
            "trajectory_csv": trajectory_csv,
            "mpnn_csv": mpnn_csv,
            "trajectory_results": trajectory_results,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hallucinate_binder")
async def api_hallucinate_binder(request: HallucinateRequest):
    """
    Run only the binder backbone hallucination step.
    Returns trajectory info and output PDB paths.
    """
    try:
        advanced_settings = _check_advanced(request.advanced_settings or dict(_default_advanced))
        design_paths = generate_directories(request.design_path)

        failure_csv = os.path.join(request.design_path, "failure_csv.csv")
        filters_json_path = os.path.join(request.design_path, "_api_filters.json")
        filters = dict(_default_filters)
        with open(filters_json_path, 'w') as f:
            json.dump(filters, f)
        generate_filter_pass_csv(failure_csv, filters_json_path)

        seed = request.seed if request.seed is not None else int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
        helicity_value = load_helicity(advanced_settings)

        design_name = request.binder_name + f"_l{request.length}_s{seed}"
        trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

        print(f"Hallucinating binder: {design_name}")
        start_time = time.time()

        trajectory = binder_hallucination(
            design_name, request.starting_pdb, request.chains,
            request.target_hotspot_residues, request.length, seed, helicity_value,
            _design_models, advanced_settings, design_paths, failure_csv,
        )

        elapsed = time.time() - start_time
        trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
        trajectory_metrics = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics.items()}
        trajectory_sequence = trajectory.get_seq(get_best=True)[0]

        return {
            "status": "success",
            "msg": "Binder hallucination完成",
            "design_name": design_name,
            "trajectory_pdb": trajectory_pdb if os.path.exists(trajectory_pdb) else None,
            "terminated": trajectory.aux["log"]["terminate"],
            "sequence": trajectory_sequence,
            "metrics": trajectory_metrics,
            "elapsed_seconds": round(elapsed, 1),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict_complex")
async def api_predict_complex(request: PredictComplexRequest):
    """
    Predict binder-target complex structure for a given sequence using AF2.
    """
    try:
        advanced_settings = _check_advanced(request.advanced_settings or dict(_default_advanced))
        filters = request.filters or dict(_default_filters)
        design_paths = generate_directories(request.design_path)

        failure_csv = os.path.join(request.design_path, "failure_csv.csv")
        filters_json_path = os.path.join(request.design_path, "_api_filters.json")
        with open(filters_json_path, 'w') as f:
            json.dump(filters, f)
        generate_filter_pass_csv(failure_csv, filters_json_path)

        clear_mem()

        prediction_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=advanced_settings.get("num_recycles_validation", 3),
            data_dir=advanced_settings.get("af_params_dir", "/app"),
            use_multimer=_multimer_validation,
            use_initial_guess=advanced_settings.get("predict_initial_guess", True),
            use_initial_atom_pos=advanced_settings.get("predict_bigbang", False),
        )

        if advanced_settings.get("predict_initial_guess") or advanced_settings.get("predict_bigbang"):
            prediction_model.prep_inputs(
                pdb_filename=request.trajectory_pdb, chain='A', binder_chain='B',
                binder_len=request.binder_length, use_binder_template=True,
                rm_target_seq=advanced_settings.get("rm_template_seq_predict", True),
                rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
                rm_template_ic=True,
            )
        else:
            prediction_model.prep_inputs(
                pdb_filename=request.target_pdb,
                chain=request.target_chain,
                binder_len=request.binder_length,
                rm_target_seq=advanced_settings.get("rm_template_seq_predict", True),
                rm_target_sc=advanced_settings.get("rm_template_sc_predict", False),
            )

        binder_sequence = re.sub("[^A-Z]", "", request.binder_sequence.upper())
        prediction_stats = {}

        for model_num in _prediction_models:
            complex_pdb = os.path.join(
                design_paths["MPNN"],
                f"{request.design_name}_model{model_num+1}.pdb",
            )
            prediction_model.predict(
                seq=binder_sequence, models=[model_num],
                num_recycles=advanced_settings.get("num_recycles_validation", 3),
                verbose=False,
            )
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"])
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2),
                'pTM': round(prediction_metrics['ptm'], 2),
                'i_pTM': round(prediction_metrics['i_ptm'], 2),
                'pAE': round(prediction_metrics['pae'], 2),
                'i_pAE': round(prediction_metrics['i_pae'], 2),
                'pdb': complex_pdb,
            }
            prediction_stats[model_num+1] = stats

            # Relax
            relaxed_pdb = os.path.join(
                design_paths["MPNN/Relaxed"],
                f"{request.design_name}_model{model_num+1}.pdb",
            )
            pr_relax(complex_pdb, relaxed_pdb)
            stats['relaxed_pdb'] = relaxed_pdb

        return {
            "status": "success",
            "msg": "复合物结构预测完成",
            "design_name": request.design_name,
            "sequence": binder_sequence,
            "models": prediction_stats,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "42001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
