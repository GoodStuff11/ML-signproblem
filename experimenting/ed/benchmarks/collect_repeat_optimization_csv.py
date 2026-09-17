#!/usr/bin/env python3
"""Concatenate the per-task CSVs of a repeat-optimization experiment into one table.

Each SLURM array task of `submit_repeat_optimization_*.sh` writes
`<ansatz>_<loss>_runs_<start>_<end>.csv` into a shared output directory. This
merges them, sorts by (ansatz, loss_type, run), checks that every
(ansatz, loss_type, run) appears exactly once, and prints a per-configuration
summary of the final losses.

Usage:
  python3 collect_repeat_optimization_csv.py <indir> <out.csv>
"""

import csv
import glob
import os
import statistics
import sys

ANSATZ_ORDER = {"trotter": 0, "exact": 1}
LOSS_ORDER = {"overlap": 0, "energy": 1}


def main(indir, out_path):
    files = sorted(glob.glob(os.path.join(indir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {indir}")

    header = None
    rows = []
    for path in files:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            file_header = next(reader, None)
            if file_header is None:
                continue
            if header is None:
                header = file_header
            elif file_header != header:
                sys.exit(f"Header mismatch in {path}:\n  {file_header}\n  {header}")
            rows.extend(row for row in reader if row)

    def sort_key(row):
        return (ANSATZ_ORDER.get(row[0], 99), LOSS_ORDER.get(row[1], 99), int(row[2]))

    rows.sort(key=sort_key)

    seen = {}
    for row in rows:
        key = (row[0], row[1], int(row[2]))
        if key in seen:
            sys.exit(f"Duplicate row for {key}")
        seen[key] = row

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Merged {len(files)} files -> {out_path} ({len(rows)} runs)\n")

    loss_col = header.index("final_loss")
    configs = {}
    for row in rows:
        configs.setdefault((row[0], row[1]), []).append(float(row[loss_col]))

    print(f"{'ansatz':8} {'loss':8} {'n':>4} {'min':>14} {'median':>14} {'max':>14} {'mean':>14}")
    for (ansatz, loss), vals in sorted(configs.items(), key=lambda kv: sort_key([kv[0][0], kv[0][1], 0])):
        print(
            f"{ansatz:8} {loss:8} {len(vals):4d} "
            f"{min(vals):14.6e} {statistics.median(vals):14.6e} "
            f"{max(vals):14.6e} {statistics.fmean(vals):14.6e}"
        )
        runs = sorted(r for (a, l, r) in seen if a == ansatz and l == loss)
        expected = list(range(1, len(runs) + 1))
        if runs != expected:
            print(f"    WARNING: run indices are not 1..{len(runs)}: {runs}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
