from pyrosetta import rosetta, init
import os


def calculate_interface_metrics(pdb_path, iface_selector="A_HL", relax_cycles=1):
    """
    Calculate binding delta G and buried SASA for a PDB structure.
    
    Args:
        pdb_path: Path to PDB file
        iface_selector: Interface selector string (default: "A_HL" for antigen A vs antibody H/L)
        relax_cycles: Number of FastRelax cycles (default: 1)
    
    Returns:
        dict: {"binding_dG": float, "buried_sasa": float} or {"error": str} if failed
    """
    try:
        # Initialize PyRosetta if not already done
        if not rosetta.basic.options.option.get_boolean_option("initialized"):
            init("-ex1 -ex2aro")
        
        # Load pose and relax
        pose = rosetta.core.import_pose.pose_from_file(pdb_path)
        scorefxn = rosetta.core.scoring.get_score_function()
        relax = rosetta.protocols.relax.FastRelax(scorefxn, relax_cycles)
        relax.apply(pose)

        # Calculate interface metrics
        iam = rosetta.protocols.analysis.InterfaceAnalyzerMover(iface_selector, False)
        iam.apply(pose)

        return {
            "binding_dG": iam.get_interface_dG(),
            "buried_sasa": iam.get_interface_delta_sasa()
        }
    except Exception as e:
        return {"error": str(e)}


def batch_calculate_interface_metrics(pdb_paths, iface_selector="A_HL", relax_cycles=1):
    """
    Calculate interface metrics for multiple PDB files.
    
    Args:
        pdb_paths: List of PDB file paths
        iface_selector: Interface selector string
        relax_cycles: Number of FastRelax cycles
    
    Returns:
        list: List of results, each containing metrics or error info
    """
    results = []
    
    for pdb_path in pdb_paths:
        result = calculate_interface_metrics(pdb_path, iface_selector, relax_cycles)
        result["pdb_file"] = pdb_path
        result["model"] = os.path.basename(os.path.dirname(pdb_path))
        result["target_id"] = os.path.splitext(os.path.basename(pdb_path))[0]
        results.append(result)
    
    return results
