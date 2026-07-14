import os
import glob
import re

logs_dir = "/home/jek354/research/ML-signproblem/experimenting/ed/logs"
log_files = [
    os.path.join(logs_dir, "system_scaling_2026-06-01_10-34-13.log"),
    os.path.join(logs_dir, "system_scaling_2026-06-01_10-40-27.log"),
    os.path.join(logs_dir, "system_scaling_2026-06-01_10-46-47.log"),
    os.path.join(logs_dir, "system_scaling_2026-06-01_10-53-13.log")
]

def get_trained_datasets(folder_set):
    if folder_set == "square_pure":
        return {"data/N=(3, 3)_3x3"}
    return set()

out_lines = []
out_lines.append("Analyzing Neural Network Hyperparameter Sweep at target U ≈ 8 (7.5 <= U <= 9.5)...")
out_lines.append("=" * 120)

for lf in log_files:
    if not os.path.exists(lf):
        out_lines.append(f"\nMissing file: {lf}")
        continue
        
    filename = os.path.basename(lf)
    with open(lf, 'r') as f:
        content = f.read()
        
    weighting = None
    u_range = None
    folder_set = None
    name = None
    
    m_w = re.search(r"Training NN with weighting scheme:\s+(\S+)", content)
    if m_w: weighting = m_w.group(1)
    
    m_u = re.search(r"Training NN with U index range:\s+(\S+)", content)
    if m_u: u_range = m_u.group(1)
    
    m_f = re.search(r"Training NN with folder set:\s+(\S+)", content)
    if m_f: folder_set = m_f.group(1)
    
    # Parse custom NN hyperparameters
    base_hidden = "default"
    embed_dim = "default"
    context_hidden = "default"
    scale_hidden = "default"
    
    m_hp = re.search(r"Training NN with architecture parameters:\s+base_hidden=([^,]+),\s+embed_dim=([^,]+),\s+context_hidden=([^,]+),\s+scale_hidden=(\S+)", content)
    if m_hp:
        base_hidden = m_hp.group(1)
        embed_dim = m_hp.group(2)
        context_hidden = m_hp.group(3)
        scale_hidden = m_hp.group(4)
        
    m_name = re.search(r"Saved neural network strategy to:\s+trained_neural_networks/trained_neural_network_(\S+)\.jld2", content)
    if m_name:
        name = m_name.group(1)
        
    trained_sets = get_trained_datasets(folder_set)
    sections = content.split("Testing on folder: ")
    
    ok_folders = []
    failed_folders = []
    untrained_results = []
    all_untrained_satisfied = True
    
    for sec in sections[1:]:
        lines = sec.strip().split('\n')
        folder = lines[0].strip()
        is_trained = folder in trained_sets
        
        table_started = False
        data_rows = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("U-va") or line_str.startswith("U-value"):
                table_started = True
                continue
            if table_started:
                if line_str.startswith("---") or line_str == "":
                    continue
                if line_str.startswith("Done."):
                    break
                parts = line_str.split()
                if len(parts) >= 13:
                    try:
                        float(parts[0])
                        data_rows.append(parts)
                    except ValueError:
                        pass
        
        if not data_rows:
            continue
            
        u8_rows = []
        for row in data_rows:
            u_val = float(row[0])
            if 7.5 <= u_val <= 9.5:
                u8_rows.append(row)
                
        if not u8_rows:
            continue
            
        max_pr = max(float(r[11]) for r in u8_rows)
        mean_pr = sum(float(r[11]) for r in u8_rows) / len(u8_rows)
        max_rb = max(float(r[12]) for r in u8_rows)
        mean_rb = sum(float(r[12]) for r in u8_rows) / len(u8_rows)
        
        folder_short = folder.replace("data/", "")
        
        if is_trained:
            continue
            
        status = "OK" if (max_pr < 1.0 and max_rb < 1.0) else "FAILED"
        if status == "FAILED":
            all_untrained_satisfied = False
            failed_folders.append(f"{folder_short} (pr={max_pr:.2f}, rb={max_rb:.2f})")
        else:
            ok_folders.append(f"{folder_short} (pr={max_pr:.2f}, rb={max_rb:.2f})")
            
        untrained_results.append({
            'folder': folder_short,
            'max_pr': max_pr,
            'mean_pr': mean_pr,
            'max_rb': max_rb,
            'mean_rb': mean_rb,
            'status': status
        })
            
    out_lines.append(f"\nFILE: {filename} | name={name} | weighting={weighting} | folder-set={folder_set}")
    out_lines.append(f"  Architecture: base={base_hidden} | embed={embed_dim} | ctx={context_hidden} | scale={scale_hidden}")
    out_lines.append(f"  All untrained satisfied? {all_untrained_satisfied}")
    out_lines.append(f"  OK datasets:     {', '.join(ok_folders) if ok_folders else 'None'}")
    out_lines.append(f"  FAILED datasets: {', '.join(failed_folders) if failed_folders else 'None'}")
    out_lines.append("  Details:")
    for res in untrained_results:
        out_lines.append(f"    - {res['folder']:<25}: pred/rand (max={res['max_pr']:.4f}, mean={res['mean_pr']:.4f}) | rand/baseline (max={res['max_rb']:.4f}, mean={res['mean_rb']:.4f}) -> {res['status']}")

with open("/home/jek354/research/ML-signproblem/experimenting/ed/hyperparam_sweep_results.txt", "w") as f:
    f.write("\n".join(out_lines))
