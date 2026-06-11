"""
BindCraft MCP Server

Exposes BindCraft protein binder design tools via the MCP (Model Context Protocol)
streamable-http transport. Internally calls the BindCraft FastAPI service.

References:
  - cardiorgan/server_cardiorgan.py for the FastMCP pattern
  - cardiorgan/server_test.py for the MCP client pattern
"""

import json
import os
import asyncio
from typing import Any, Dict, List, Optional

import requests
from fastmcp import FastMCP

API_URL = os.environ.get("BINDCRAFT_API_URL", "http://localhost:42001")

# Create MCP server instance
mcp = FastMCP("BindCraftTools")


# ==============================================================================
# Synchronous HTTP helper
# ==============================================================================

def sync_post(url: str, params: Dict[str, Any], timeout: int = 3600) -> Dict[str, Any]:
    """Synchronous POST request to the BindCraft API server."""
    try:
        response = requests.post(url, json=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API request failed with status {response.status_code}: {response.text[:500]}"
            }
    except requests.exceptions.Timeout:
        return {"error": "Request timeout (design may still be running on the server)"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to API server at {url}. Is the BindCraft API running?"}
    except Exception as e:
        return {"error": f"Request error: {str(e)}"}


def sync_get(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Synchronous GET request."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Health check failed with status {response.status_code}"}
    except Exception as e:
        return {"error": f"Health check error: {str(e)}"}


# ==============================================================================
# Tool 001: Health check
# ==============================================================================

@mcp.tool(
    meta={
        "scp_properties": {
            "type": "sync",
            "limit": 10
        }
    }
)
async def bindcraft_health() -> Dict[str, Any]:
    """Check if the BindCraft API server is healthy and GPU is available.

    Returns:
        status: healthy/unhealthy
        gpu_available: whether GPU devices are detected
        gpu_devices: list of GPU device names
    """
    url = f"{API_URL}/health"
    result = await asyncio.to_thread(sync_get, url=url)
    return result


# ==============================================================================
# Tool 002: Run full binder design
# ==============================================================================

@mcp.tool(
    meta={
        "scp_properties": {
            "type": "sync",
            "limit": 1
        }
    }
)
async def run_binder_design(
    target_settings: Dict[str, Any],
    advanced_settings: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the complete BindCraft de novo protein binder design pipeline.

    This performs the full workflow: backbone hallucination, relaxation, interface scoring,
    ProteinMPNN sequence redesign, AlphaFold2 complex prediction, and filter-based selection.

    Args:
        target_settings: Target protein settings dict with keys:
            - starting_pdb (str, required): Path to target protein PDB file
            - chains (str): Target chain ID(s), e.g. "A"
            - target_hotspot_residues (str): Comma-separated hotspot residue numbers
            - lengths (list[int]): Min and max binder lengths, e.g. [65, 150]
            - binder_name (str): Name prefix for designs
            - number_of_final_designs (int): Stop after this many accepted designs
            - design_path (str): Output directory path
        advanced_settings: Optional advanced protocol settings (algorithm, iterations, loss weights, etc.)
        filters: Optional filter thresholds for design selection

    Returns:
        status: success/error
        msg: Summary message
        accepted_designs: Number of designs that passed all filters
        accepted_names: List of accepted design names
        design_path: Output directory
        final_csv: Path to final design statistics CSV
        elapsed: Human-readable elapsed time
    """
    url = f"{API_URL}/api/run_design"
    params = {
        "target_settings": target_settings,
        "advanced_settings": advanced_settings,
        "filters": filters,
    }
    result = await asyncio.to_thread(sync_post, url=url, params=params, timeout=86400)
    return result


# ==============================================================================
# Tool 003: Hallucinate binder backbone only
# ==============================================================================

@mcp.tool(
    meta={
        "scp_properties": {
            "type": "sync",
            "limit": 2
        }
    }
)
async def hallucinate_binder(
    starting_pdb: str,
    length: int = 100,
    chains: str = "A",
    target_hotspot_residues: str = "",
    binder_name: str = "binder",
    seed: Optional[int] = None,
    design_path: str = "/workspace/designs/default",
    advanced_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run only the binder backbone hallucination step (no MPNN redesign or validation).

    Uses AlphaFold2 backpropagation via ColabDesign to hallucinate a binder backbone
    against the target protein at the specified hotspot residues.

    Args:
        starting_pdb (str): Path to target protein PDB file
        length (int): Desired binder length in residues
        chains (str): Target chain ID(s)
        target_hotspot_residues (str): Comma-separated hotspot residues
        binder_name (str): Name prefix for the design
        seed (int, optional): Random seed for reproducibility
        design_path (str): Output directory
        advanced_settings (dict, optional): Protocol overrides

    Returns:
        status: success/error
        design_name: Name of the generated design
        trajectory_pdb: Path to the hallucinated PDB file
        sequence: Binder amino acid sequence
        metrics: AF2 confidence metrics (plddt, ptm, i_ptm, pae, i_pae)
    """
    url = f"{API_URL}/api/hallucinate_binder"
    params = {
        "binder_name": binder_name,
        "starting_pdb": starting_pdb,
        "chains": chains,
        "target_hotspot_residues": target_hotspot_residues,
        "length": length,
        "seed": seed,
        "design_path": design_path,
        "advanced_settings": advanced_settings,
    }
    result = await asyncio.to_thread(sync_post, url=url, params=params, timeout=7200)
    return result


# ==============================================================================
# Tool 004: Predict complex structure
# ==============================================================================

@mcp.tool(
    meta={
        "scp_properties": {
            "type": "sync",
            "limit": 2
        }
    }
)
async def predict_binder_complex(
    binder_sequence: str,
    target_pdb: str,
    binder_length: int,
    trajectory_pdb: str,
    target_chain: str = "A",
    design_path: str = "/workspace/designs/default",
    design_name: str = "prediction",
    advanced_settings: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predict the 3D structure of a binder-target complex using AlphaFold2.

    Given a binder amino acid sequence and a target protein, predicts the complex
    structure using AF2 with masked templates.

    Args:
        binder_sequence (str): Amino acid sequence of the binder (one-letter codes)
        target_pdb (str): Path to target protein PDB file
        binder_length (int): Length of the binder in residues
        trajectory_pdb (str): Path to the trajectory/hallucination PDB for template
        target_chain (str): Target chain ID
        design_path (str): Output directory
        design_name (str): Name for this prediction
        advanced_settings (dict, optional): AF2 prediction settings
        filters (dict, optional): Per-model AF2 filter thresholds

    Returns:
        status: success/error
        models: Dict of per-model statistics and PDB paths
        sequence: The predicted binder sequence
    """
    url = f"{API_URL}/api/predict_complex"
    params = {
        "binder_sequence": binder_sequence,
        "target_pdb": target_pdb,
        "target_chain": target_chain,
        "binder_length": binder_length,
        "trajectory_pdb": trajectory_pdb,
        "design_path": design_path,
        "design_name": design_name,
        "advanced_settings": advanced_settings,
        "filters": filters,
    }
    result = await asyncio.to_thread(sync_post, url=url, params=params, timeout=7200)
    return result


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "32210"))
    print("\n" + "=" * 70)
    print(f"MCP Server - BindCraftTools (API: {API_URL})")
    print("=" * 70)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
