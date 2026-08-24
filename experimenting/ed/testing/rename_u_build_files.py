"""
Script to rename .jld2 files in data_h5_fixed to remove `u_build` labels,
except when there are duplicates and when growing from num_exponentials=1 to 2.
"""

import os
import sys
import shutil

ROOT = "/home/jek354/research/data/new_data/data_h5_fixed"

def plan_renames(root=ROOT):
    actions = []
    
    for dirpath, _, filenames in os.walk(root):
        all_jld2 = set(f for f in filenames if f.endswith(".jld2"))
        u_build_files = sorted(f for f in filenames if f.endswith(".jld2") and "u_build" in f)
        
        for f in u_build_files:
            # Determine bare name
            if "_u_build_" in f:
                bare = f.replace("_u_build_", "_")
            else:
                bare = f.replace("_u_build", "")
            
            src_path = os.path.join(dirpath, f)
            dst_path = os.path.join(dirpath, bare)
            
            # Check if this is a num_exponentials=2 file and a duplicate bare file exists
            is_growth = "num_exponentials=2" in f
            bare_exists = bare in all_jld2
            
            if is_growth and bare_exists:
                # Keep u_build for duplicate growth files
                actions.append({
                    "action": "KEEP",
                    "reason": "Duplicate num_exponentials=2 growth file",
                    "folder": os.path.relpath(dirpath, root),
                    "src": f,
                    "dst": f,
                    "src_path": src_path,
                    "dst_path": src_path
                })
            elif bare_exists:
                # Replace older bare file with newer u_build file
                actions.append({
                    "action": "OVERWRITE_BARE",
                    "reason": "Duplicate num_exponentials=1 file (replace bare with u_build)",
                    "folder": os.path.relpath(dirpath, root),
                    "src": f,
                    "dst": bare,
                    "src_path": src_path,
                    "dst_path": dst_path
                })
            else:
                # Standard rename
                actions.append({
                    "action": "RENAME",
                    "reason": "Remove u_build suffix",
                    "folder": os.path.relpath(dirpath, root),
                    "src": f,
                    "dst": bare,
                    "src_path": src_path,
                    "dst_path": dst_path
                })
                
    return actions

def main():
    dry_run = "--execute" not in sys.argv
    actions = plan_renames()
    
    counts = {}
    for a in actions:
        counts[a["action"]] = counts.get(a["action"], 0) + 1
        
    print(f"Total u_build files found: {len(actions)}")
    for act, cnt in counts.items():
        print(f"  {act}: {cnt}")
        
    if dry_run:
        print("\n[DRY RUN] No files modified. Pass --execute to apply changes.")
        print("\nSample actions:")
        for a in actions[:10]:
            print(f"  [{a['action']}] {a['folder']}/{a['src']} -> {a['dst']}")
    else:
        print("\n[EXECUTING] Applying renames...")
        for a in actions:
            if a["action"] == "KEEP":
                continue
            elif a["action"] in ("RENAME", "OVERWRITE_BARE"):
                # Move src to dst (replacing dst if exists)
                os.replace(a["src_path"], a["dst_path"])
        print("Done!")

if __name__ == "__main__":
    main()
