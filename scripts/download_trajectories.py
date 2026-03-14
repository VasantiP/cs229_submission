import time
import requests
import pandas as pd
import os
from google.cloud import storage
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================
CHECK_PDBID_EXISTENCE = True  # Set to False to skip existence check (faster for testing, but may cause duplicates)
INPUT_CSV = "targets.csv"
BUCKET_NAME = "cs229-central"
BASE_DIR = "data"
MAX_WORKERS = 4  # Adjust based on VM's network capacity

# Initialize GCS Client
storage_client = storage.Client()

def sanitize(text):
    """Replaces dots and spaces with underscores for tool compatibility."""
    if pd.isna(text): 
        return "unknown"
    # Strip whitespace, replace problematic characters
    clean = str(text).strip().replace('.', '_').replace(' ', '_')
    # Consolidate multiple underscores (e.g., "PDB . 1" -> "PDB___1" -> "PDB_1")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean

def pdbid_exists_in_gcs(pdbid):
    if not CHECK_PDBID_EXISTENCE:
        return False

    """Checks if any file with the given PDB ID exists in the bucket."""
    bucket = storage_client.bucket(BUCKET_NAME)
    prefix = f"{BASE_DIR}/"

    directories = ['trajectories', 'topologies', 'models']
    for dir in directories:
        dir_prefix = f"{prefix}{dir}/"
        blobs = bucket.list_blobs(prefix=dir_prefix)
    
        for blob in blobs:
            if f"~{pdbid}~" in blob.name:
                return True

    return False    

def file_exists_in_gcs(gcs_path):
    """Checks if a blob already exists in the bucket to skip download."""
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket.blob(gcs_path).exists()

def download_file(url, gcs_path, pbar_desc):
    """Streams a file from URL to GCS and validates final size."""
    start_time = time.time()
    try:
        if not url or pd.isna(url) or "REPLACE" in str(url):
            return False, 0, 0, "Missing/Placeholder URL"
            
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        with requests.get(url, stream=True, timeout=45) as r:
            r.raise_for_status()
            expected_size = int(r.headers.get('content-length', 0))
            
            # Individual progress bar for the file
            with tqdm(total=expected_size, unit='B', unit_scale=True, desc=pbar_desc, leave=False) as pbar:
                with blob.open("wb", content_type='application/octet-stream') as gcs_file:
                    for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                        if chunk:
                            gcs_file.write(chunk)
                            pbar.update(len(chunk))
            
            # Metadata reload for size validation
            blob.reload()
            actual_size = blob.size
            duration = time.time() - start_time
            
            if expected_size > 0 and actual_size != expected_size:
                return False, duration, actual_size, f"Truncated: {actual_size}/{expected_size}"
                
        return True, duration, actual_size, "Success"
    except Exception as e:
        return False, time.time() - start_time, 0, str(e)

def process_gpcr_package(traj):
    """Handles the triple download (XTC, PSF, PDB) for a single entry."""
    raw_pdbid = str(traj['pdbid'])
    clean_pdbid = sanitize(raw_pdbid)
    clean_prot = sanitize(traj['protein'])
    state = traj.get('state', 'unknown')
    
    pkg_results = {
        'original_pdbid': raw_pdbid, 
        'sanitized_pdbid': clean_pdbid,
        'protein': clean_prot, 
        'state': state,
        'all_clear': True,
        'total_mb': 0,
        'total_sec': 0
    }

    if pdbid_exists_in_gcs(clean_pdbid):
        tqdm.write(f"Skipping PDB ID: {clean_pdbid} (Already exists)")
        return pkg_results

    file_name_prefix = f"{clean_prot}~{clean_pdbid}~{state}"
    # Only create a file if exists in traj 
    files = []
    if 'trajectory_url' in traj and not pd.isna(traj['trajectory_url']):
        traj_url = traj.get('trajectory_url')
        traj_suffix = traj_url.split('.')[-1]
        traj_file_name = traj_url.replace(f"/trajectory", f"_trajectory").split('/')[-1].replace(f".{traj_suffix}", "")
#        traj_file_name = traj_url.split('/')[-1].replace(f".{traj_suffix}", "")
        traj_file_name = f"{file_name_prefix}~{traj_file_name}.{traj_suffix}"
        files.append((traj_suffix, traj_url, f"{BASE_DIR}/trajectories/{traj_file_name}"))
    if 'topology_url' in traj and not pd.isna(traj['topology_url']):
        top_url = traj.get('topology_url')
        top_suffix = top_url.split('.')[-1]
        top_file_name = top_url.replace(f"/topology", f"_topology").split('/')[-1].replace(f".{top_suffix}", "")
#        top_file_name = top_url.split('/')[-1].replace(f".{top_suffix}", "")
        top_file_name = f"{file_name_prefix}~{top_file_name}.{top_suffix}"
        files.append((top_suffix, top_url, f"{BASE_DIR}/topologies/{top_file_name}"))
    if 'model_url' in traj and not pd.isna(traj['model_url']):
        model_url = traj.get('model_url')
        model_suffix = model_url.split('.')[-1]
        model_file_name = model_url.replace(f"/structure", f"_structure").split('/')[-1].replace(f".{model_suffix}", "")
        model_file_name = f"{file_name_prefix}~{model_file_name}.{model_suffix}"
        files.append((model_suffix, model_url, f"{BASE_DIR}/models/{model_file_name}"))
    
    for suffix, url, path in files:
        file_name = path.split('/')[-1]
        # Check for existence before attempting download
        if file_exists_in_gcs(path):
            pkg_results[f'{suffix}_ok'] = True
            tqdm.write(f"Skipping {suffix.upper()}: {file_name} (Exists)")
            continue

        success, duration, size_bytes, msg = download_file(url, path, f"  {suffix.upper()}: {file_name}")
        
        pkg_results[f'{suffix}_ok'] = success
        pkg_results[f'{suffix}_sec'] = round(duration, 2)
        pkg_results[f'{suffix}_mb'] = round(size_bytes / (1024**2), 2)
        pkg_results['total_mb'] += pkg_results[f'{suffix}_mb']
        pkg_results['total_sec'] += duration
        
        if not success:
            pkg_results['all_clear'] = False
            tqdm.write(f"[FAIL] {file_name} {suffix}: {msg}")
            
    return pkg_results

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please ensure your CSV is in the same folder.")
    else:
        # Load and start
        df_input = pd.read_csv(INPUT_CSV)
        trajectories = df_input.to_dict('records')
        
        print(f"Loaded {len(trajectories)} entries. Starting parallel sync to gs://{BUCKET_NAME}/{BASE_DIR}/")
        
        overall_start = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_gpcr_package, t) for t in trajectories]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Job Progress"):
                res = future.result()
                results.append(res)
                if res['all_clear']:
                    tqdm.write(f"Synced {res['sanitized_pdbid']} ({res['total_mb']:.1f} MB)")

        # Summary Reporting
        total_wall_time = time.time() - overall_start
        df_out = pd.DataFrame(results)
        df_out.to_csv('final_sync_log.csv', index=False)
        
        total_data_gb = df_out['total_mb'].sum() / 1024
        avg_speed_mbps = (df_out['total_mb'].sum() * 8) / total_wall_time

        print("\n" + "="*60)
        print("SYNC COMPLETE")
        print(f"Total Time: {total_wall_time/60:.2f} minutes")
        print(f"Data Transferred: {total_data_gb:.2f} GB")
        print(f"Effective Session Speed: {avg_speed_mbps:.2f} Mbps")
        print(f"Success Rate: {df_out['all_clear'].sum()} / {len(df_out)}")
        print("="*60)
