"""
Comprehensive feature extraction for MD trajectories.

Extracts TIER 1 features at multiple trajectory lengths:
- Scalar features (RMSD, Rg, TM3-TM6)
- TICA features (projections + eigenvalues)
- At 20%, 30%, 50% trajectory length

Run this script on GCS/Vertex AI to process all trajectories once.
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# TICA
try:
    import pyemma.coordinates as coor
    HAS_PYEMMA = True
except ImportError:
    print("PyEMMA not found, TICA features will be skipped")
    HAS_PYEMMA = False

# For robust NaN handling
import warnings
warnings.filterwarnings('ignore')
np.bool = bool  # PyEMMA compatibility


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Trajectory lengths to extract
EARLY_FRACS = [0.60, 0.70, 0.80, 0.90]  

# TICA parameters 
TICA_LAG = 10
TICA_DIM = 5

# Atom selection for alignment
ATOM_SEL = "protein and name CA"

# Output structure
OUTPUT_DIR = Path("data/processed_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_scalar_features(u, early_frac):
    """
    Extract scalar time series: RMSD, Rg, TM3-TM6 distance.
    
    Returns:
        np.array of shape (n_early_frames, 3)
    """
    ca = u.select_atoms(ATOM_SEL)
    n_frames = len(u.trajectory)
    n_early = int(n_frames * early_frac)
    
    # Reference structure (first frame)
    u.trajectory[0]
    ref_coords = ca.positions.copy()
    
    features = []
    for ts in u.trajectory[:n_early]:
        # Align to reference
        align.alignto(ca, ca, select=ATOM_SEL, weights="mass")
        
        # RMSD
        rmsd = rms.rmsd(ca.positions, ref_coords, superposition=False)
        
        # Radius of gyration
        rg = ca.radius_of_gyration()
        
        # TM3-TM6 distance (robust calculation)
        tm_dist = compute_tm_distance_robust(u)
        
        features.append([rmsd, rg, tm_dist])
    
    return np.array(features)


def compute_tm_distance_robust(u):
    """
    Compute TM3-TM6 distance with fallback strategies.
    
    Returns distance or NaN if all strategies fail.
    """
    ca = u.select_atoms("protein and name CA")
    
    # Strategy 1: Try standard GPCR TM helix positions
    # TM3 typically around residue 100-150, TM6 around 250-300
    # Adjust these ranges based on your specific proteins
    try:
        # Use sequence positions (adjust for your proteins)
        n_res = len(ca)
        tm3_start = int(n_res * 0.25)  # Roughly 1/4 through
        tm3_end = int(n_res * 0.35)
        tm6_start = int(n_res * 0.65)  # Roughly 2/3 through
        tm6_end = int(n_res * 0.75)
        
        tm3 = ca[tm3_start:tm3_end]
        tm6 = ca[tm6_start:tm6_end]
        
        if len(tm3) > 0 and len(tm6) > 0:
            dist = np.linalg.norm(
                tm3.center_of_mass() - tm6.center_of_mass()
            )
            return dist
    except Exception as e:
        pass
    
    # Strategy 2: Use overall protein ends as proxy
    # (Less accurate but always works)
    try:
        n_ca = len(ca)
        n_term = ca[:n_ca//3].center_of_mass()
        c_term = ca[2*n_ca//3:].center_of_mass()
        dist = np.linalg.norm(n_term - c_term)
        return dist
    except Exception as e:
        pass
    
    # All strategies failed
    return np.nan


def compute_tica_features(u, early_frac):
    """
    Extract TICA features: eigenvalues and projections.
    
    Returns:
        eigenvalues: np.array of shape (TICA_DIM,)
        projections: np.array of shape (n_early_frames, TICA_DIM)
    """
    if not HAS_PYEMMA:
        return None, None
    
    ca = u.select_atoms(ATOM_SEL)
    n_frames = len(u.trajectory)
    n_early = int(n_frames * early_frac)
    
    # Reference structure
    u.trajectory[0]
    ref_coords = ca.positions.copy()
    
    # Extract aligned coordinates
    coords = []
    for ts in u.trajectory[:n_early]:
        align.alignto(ca, ca, select=ATOM_SEL, weights="mass")
        coords.append(ca.positions.flatten())
    
    coords = np.array(coords)
    
    # Check if enough frames for TICA
    if len(coords) <= TICA_LAG + 1:
        return None, None
    
    try:
        # Fit TICA (per-trajectory)
        tica = coor.tica(coords, lag=TICA_LAG, dim=TICA_DIM)
        
        # Get eigenvalues
        eigenvalues = tica.eigenvalues[:TICA_DIM]
        
        # Get projections (time series)
        projections = tica.get_output()[0]  # Shape: (n_frames, TICA_DIM)
        
        return eigenvalues, projections
        
    except Exception as e:
        print(f"    TICA failed: {e}")
        return None, None


def process_trajectory(traj_path, top_path, traj_id, output_base):
    """
    Process one trajectory and extract all features at all lengths.
    
    Args:
        traj_path: Path to trajectory file (.xtc)
        top_path: Path to topology file (.psf/.pdb)
        traj_id: Unique trajectory identifier
        output_base: Base output directory
    
    Returns:
        dict with metadata about extracted features
    """
    try:
        # Load trajectory
        u = mda.Universe(str(top_path), str(traj_path))
        n_frames_total = len(u.trajectory)
        
        results = {
            'traj_id': traj_id,
            'n_frames_total': n_frames_total,
            'success': True,
            'error': None
        }
        
        # Process each trajectory length
        for early_frac in EARLY_FRACS:
            frac_str = f"{int(early_frac*100):02d}pct"
            n_early = int(n_frames_total * early_frac)
            
            results[f'n_frames_{frac_str}'] = n_early
            
            # Create output directories
            scalar_dir = output_base / f"features_{frac_str}" / "scalar"
            tica_proj_dir = output_base / f"features_{frac_str}" / "tica" / "projections"
            tica_eig_dir = output_base / f"features_{frac_str}" / "tica" / "eigenvalues"
            
            for d in [scalar_dir, tica_proj_dir, tica_eig_dir]:
                d.mkdir(parents=True, exist_ok=True)
            
            # Extract scalar features
            try:
                scalar_features = compute_scalar_features(u, early_frac)
                np.save(scalar_dir / f"{traj_id}.npy", scalar_features)
                results[f'scalar_{frac_str}'] = True
            except Exception as e:
                print(f"    Scalar features failed at {frac_str}: {e}")
                results[f'scalar_{frac_str}'] = False
            
            # Extract TICA features
            try:
                eigenvalues, projections = compute_tica_features(u, early_frac)
                
                if eigenvalues is not None:
                    np.save(tica_eig_dir / f"{traj_id}.npy", eigenvalues)
                    np.save(tica_proj_dir / f"{traj_id}.npy", projections)
                    results[f'tica_{frac_str}'] = True
                else:
                    results[f'tica_{frac_str}'] = False
                    
            except Exception as e:
                print(f"    TICA features failed at {frac_str}: {e}")
                results[f'tica_{frac_str}'] = False
        
        return results
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return {
            'traj_id': traj_id,
            'success': False,
            'error': str(e)
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, 
                       default="data/processed/data_processed_v3_base_dataset_deduped.csv",
                       help="CSV with trajectory metadata")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip trajectories that already have features extracted")
    parser.add_argument("--gcs_mount", type=str, 
                       default="/home/jupyter/gcs_mount",
                       help="GCS FUSE mount point (or local path with trajectory files)")
    parser.add_argument("--output_dir", type=str,
                       default="data/processed_v4",
                       help="Output directory for features")
    parser.add_argument("--n_trajectories", type=int, default=None,
                       help="Number of trajectories to process (None = all)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE FEATURE EXTRACTION")
    print("="*80)
    
    # Load metadata
    df = pd.read_csv(args.metadata_csv)
    print(f"\nLoaded {len(df)} trajectories")
    
    if args.n_trajectories:
        df = df.head(args.n_trajectories)
        print(f"Processing first {len(df)} trajectories")
    
    # Setup output
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each trajectory
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        receptor = row['receptor']
        simID = row.get('simID', row.get('sim_id'))
        rep = row.get('rep', 1)
        
        traj_id = f"{receptor.replace('~', '_')}_sim{simID}_rep{rep}"
        
        print(f"\n[{idx+1}/{len(df)}] {receptor[:50]}...")
        
        # Check if already processed (if skip_existing is enabled)
        if args.skip_existing:
            # Check if all expected files exist
            all_exist = True
            for early_frac in EARLY_FRACS:
                frac_str = f"{int(early_frac*100):02d}pct"
                scalar_file = output_base / f"features_{frac_str}" / "scalar" / f"{traj_id}.npy"
                tica_eig_file = output_base / f"features_{frac_str}" / "tica" / "eigenvalues" / f"{traj_id}.npy"

                if not scalar_file.exists() or not tica_eig_file.exists():
                    all_exist = False
                    break

            if all_exist:
                print(f"  ✓ Already processed - skipping")
                results.append({
                    'traj_id': traj_id,
                    'receptor': receptor,
                    'simID': simID,
                    'rep': rep,
                    'label': row.get('label', None),
                    'success': True,
                    'skipped': True
                })
                continue

        # Get file paths from GCS mount
        # Files are at: /gcs_mount/data/trajectories/*.xtc
        #               /gcs_mount/data/topologies/*.psf
        
        gcs_mount = Path(args.gcs_mount)
        
        # Construct file paths based on GCS structure
        # Pattern: {receptor}~{pdb}~{state}~{id}_trj_{simID}.xtc
        traj_file = row.get('traj_file', '')
        top_file = row.get('top_file', '')
        
        if traj_file:
            traj_path = gcs_mount / "data" / "trajectories" / traj_file
        else:
            print(f"  ✗ No trajectory file in metadata")
            results.append({
                'traj_id': traj_id,
                'success': False,
                'error': 'No traj_file in metadata'
            })
            continue
        
        if top_file:
            top_path = gcs_mount / "data" / "topologies" / top_file
        else:
            print(f"  ✗ No topology file in metadata")
            results.append({
                'traj_id': traj_id,
                'success': False,
                'error': 'No top_file in metadata'
            })
            continue
        
        if not traj_path.exists() or not top_path.exists():
            print(f"  ✗ Files not found: {traj_path.name}, {top_path.name}")
            results.append({
                'traj_id': traj_id,
                'success': False,
                'error': 'Files not found'
            })
            continue
        
        # Process trajectory
        result = process_trajectory(traj_path, top_path, traj_id, output_base)
        result.update({
            'receptor': receptor,
            'simID': simID,
            'rep': rep,
            'label': row.get('label', None)
        })
        results.append(result)
    
    # Save metadata
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_base / "extraction_metadata.csv", index=False)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    
    n_success = results_df['success'].sum()
    n_skipped = results_df.get('skipped', pd.Series([False]*len(results_df))).sum()
    n_processed = n_success - n_skipped

    print(f"\nSuccessfully processed: {n_success}/{len(results_df)} trajectories")
    if n_skipped > 0:
        print(f"  Already existed (skipped): {n_skipped}")
        print(f"  Newly processed: {n_processed}")

    if n_success < len(results_df):
        print(f"\nFailed trajectories:")
        failed = results_df[~results_df['success']]
        for _, row in failed.iterrows():
            print(f"  - {row['traj_id']}: {row.get('error', 'Unknown error')}")
    
    print(f"\nFeatures saved to: {output_base}")

if __name__ == "__main__":
    main()