# AI GENERATED FILE  
# SPDX-License-Identifier: Unlicense
 
#/usr/bin/env python3
import os
import sys
import subprocess
import hashlib
import shutil
import glob

def compute_sha256(filepath):
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_downloader():
    """Returns 'aria2c' if available, otherwise 'wget'."""
    if shutil.which("aria2c"):
        return "aria2c"
    elif shutil.which("wget"):
        return "wget"
    else:
        print("Error: Neither aria2c nor wget is installed.")
        sys.exit(1)

def run_extraction(filename, extract_dir):
    """Extracts the archive into extract_dir."""
    os.makedirs(extract_dir, exist_ok=True)
    try:
        if filename.endswith(".tar.gz"):
            subprocess.run(["tar", "-xzf", filename, "-C", extract_dir], check=True, stderr=subprocess.PIPE)
        elif filename.endswith(".zip"):
            subprocess.run(["unzip", "-q", "-o", filename, "-d", extract_dir], check=True, stderr=subprocess.PIPE)
        else:
            print(f"Unknown archive format: {filename}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Extraction failed: {e}")
        return False

def flatten_directory(extract_dir, target_dir):
    """If extraction resulted in a single folder, move its contents up."""
    contents = os.listdir(extract_dir)
    if len(contents) == 1:
        inner_path = os.path.join(extract_dir, contents[0])
        if os.path.isdir(inner_path):
            for item in os.listdir(inner_path):
                shutil.move(os.path.join(inner_path, item), target_dir)
            return
    # Otherwise, move everything from extract_dir to target_dir
    for item in contents:
        shutil.move(os.path.join(extract_dir, item), target_dir)

def process_dataset(dataset_dir):
    dir_name = os.path.basename(dataset_dir)
    url_file = os.path.join(dataset_dir, "url.txt")
    hash_file = os.path.join(dataset_dir, "hash.txt")
    
    if not os.path.exists(url_file):
        return

    with open(url_file, "r") as f:
        url = f.read().strip()
    
    filename = url.split("/")[-1]
    archive_path = os.path.join(dataset_dir, filename)
    
    if not os.path.exists(hash_file):
        print(f"[{dir_name}] Skipping: hash.txt is missing. Expected pre-existing hashes.")
        return
        
    with open(hash_file, "r") as f:
        expected_hash = f.read().strip()

    print(f"\n--- Processing {dir_name} ---")

    def get_graph_file():
        candidates = glob.glob(os.path.join(dataset_dir, "*.mtx")) + glob.glob(os.path.join(dataset_dir, "*.edges"))
        # Filter out auxiliary files like sources
        candidates = [c for c in candidates if not c.endswith("_sources.mtx")]
        if not candidates:
            return None
        return max(candidates, key=os.path.getsize)

    # Stage 1: Pre-check verification
    graph_file = get_graph_file()
    if graph_file:
        print(f"[{dir_name}] Found existing graph file: {os.path.basename(graph_file)}. Verifying hash...")
        actual_hash = compute_sha256(graph_file)
        if actual_hash == expected_hash:
            print(f"[{dir_name}] Hash matches. Dataset is ready.")
            return
        else:
            print(f"[{dir_name}] Hash mismatch! Deleting corrupt files...")
            os.remove(graph_file)
            if os.path.exists(archive_path):
                os.remove(archive_path)

    downloader = get_downloader()
    max_retries = 2
    
    for attempt in range(1, max_retries + 1):
        print(f"[{dir_name}] Download attempt {attempt} using {downloader}...")
        
        # Stage 2: Download
        os.chdir(dataset_dir)
        try:
            if downloader == "aria2c":
                subprocess.run(["aria2c", "-c", "-x", "10", "-s", "10", url, "-o", filename], check=True)
            elif downloader == "wget":
                subprocess.run(["wget", "-c", "-q", "--show-progress", url, "-O", filename], check=True)
        except subprocess.CalledProcessError:
            print(f"[{dir_name}] Downloader process exited with error.")
            
        if not os.path.exists(filename):
            print(f"[{dir_name}] Failed to create archive file.")
            continue
            
        # Stage 3: Extraction & Native Check
        print(f"[{dir_name}] Extracting...")
        extract_dir = os.path.join(dataset_dir, ".tmp_extract")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
            
        success = run_extraction(filename, extract_dir)
        if not success:
            print(f"[{dir_name}] Extraction failed. Archive is likely corrupted. Deleting archive...")
            os.remove(filename)
            if os.path.exists(filename + ".aria2"):
                os.remove(filename + ".aria2")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            continue
            
        # Stage 4: Copy & Flatten
        flatten_directory(extract_dir, dataset_dir)
        shutil.rmtree(extract_dir)
        
        # Stage 5: Post-Extraction Verification
        graph_file = get_graph_file()
        if not graph_file:
            print(f"[{dir_name}] Error: No .mtx or .edges file found after extraction.")
            continue
            
        print(f"[{dir_name}] Verifying hash of extracted file...")
        actual_hash = compute_sha256(graph_file)
        if actual_hash == expected_hash:
            print(f"[{dir_name}] Successfully downloaded, extracted, and verified!")
            return
        else:
            print(f"[{dir_name}] Hash mismatch after extraction! Deleting and retrying...")
            os.remove(graph_file)
            os.remove(archive_path)
            
    print(f"[{dir_name}] Failed to obtain a valid dataset after {max_retries} attempts.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for item in sorted(os.listdir(base_dir)):
        dataset_dir = os.path.join(base_dir, item)
        if os.path.isdir(dataset_dir):
            process_dataset(dataset_dir)

if __name__ == "__main__":
    main()
