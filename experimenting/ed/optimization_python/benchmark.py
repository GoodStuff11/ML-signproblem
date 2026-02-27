import os
import sys
import time
import numpy as np
import h5py
import scipy.sparse as sp
from ed_optimization import adjoint_loss_and_grad

def load_julia_data(filename):
    with h5py.File(filename, 'r') as f:
        # Load matrices
        # In JLD2 representations of SparseMatrixCSC, they are typically stored with 
        # colptr, rowval, nzval. 
        # Since we use `ops` which is an array of sparse matrices:
        ops_refs = f['ops'] # array of object references
        
        ops = []
        for ref in ops_refs:
            # dereference the sparse matrix struct void object
            op_data = f[ref][()]
            
            # Extract elements from the struct void: m, n, colptr, rowval, nzval
            # The structure from jld2 is (m, n, <ref to colptr>, <ref to rowval>, <ref to nzval>)
            colptr_ref = op_data[2]
            rowval_ref = op_data[3]
            nzval_ref = op_data[4]
            
            colptr = f[colptr_ref][:]
            rowval = f[rowval_ref][:]
            nzval = f[nzval_ref][:]
            
            # Note JLD2 is 1-indexed, convert to 0-indexed
            # Ensure proper numeric types for scipy
            mat = sp.csc_matrix((np.array(nzval, dtype=np.complex128), 
                                 np.array(rowval - 1, dtype=np.int32), 
                                 np.array(colptr - 1, dtype=np.int32)), 
                                shape=(f['dim'][()], f['dim'][()]))
            ops.append(mat)
                
        t_vals = np.array(f['t_vals'][:], dtype=np.float64)
        
        # JLD2 sometimes saves complex arrays as void type arrays with 're' and 'im' fields
        state1_raw = f['state1'][:]
        state2_raw = f['state2'][:]
        
        if state1_raw.dtype.names and 're' in state1_raw.dtype.names:
            state1 = state1_raw['re'] + 1j * state1_raw['im']
        else:
            state1 = np.array(state1_raw, dtype=np.complex128)
            
        if state2_raw.dtype.names and 're' in state2_raw.dtype.names:
            state2 = state2_raw['re'] + 1j * state2_raw['im']
        else:
            state2 = np.array(state2_raw, dtype=np.complex128)
        
        return t_vals, ops, state1, state2

def run_benchmark():
    file_path = "benchmark_data.jld2"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Run the julia export script first.")
        sys.exit(1)
        
    t_vals, ops, state1, state2 = load_julia_data(file_path)
    dim = len(state1)
    print(f"Loaded Data: N_ops={len(ops)}, dim={dim}")
    
    # Python uses CSR often for fast mult
    ops = [op.tocsr() for op in ops]

    # Warmup
    adjoint_loss_and_grad(t_vals, ops, state2, state1, antihermitian=False)

    print("Starting Benchmark (Python SciPy)...")
    times = []
    
    for _ in range(10):
        t0 = time.time()
        loss, grad = adjoint_loss_and_grad(t_vals, ops, state2, state1, antihermitian=False)
        t1 = time.time()
        times.append(t1 - t0)
        
    times = np.array(times)
    
    print(f"Python Adjoint Gradient Time (N={len(t_vals)}, dim={dim}):")
    print(f"  Mean: {np.mean(times):.6f} s")
    print(f"  Std:  {np.std(times):.6f} s")
    print(f"  Min:  {np.min(times):.6f} s")

    print("\nVerifying numerical accuracy against Julia values...")
    try:
        with h5py.File(file_path, 'r') as f:
            j_loss = f['exact_loss'][()]
            j_grad = f['exact_grad'][:]
            
        loss_diff = abs(j_loss - loss)
        grad_diff = np.max(np.abs(j_grad - grad))
        grad_rel = grad_diff / np.max(np.abs(j_grad)) if np.max(np.abs(j_grad)) > 1e-12 else grad_diff
        
        print(f"Julia Loss:  {j_loss:.8e}")
        print(f"Python Loss: {loss:.8e}")
        print(f"Absolute Loss Difference: {loss_diff:.4e}")
        
        print(f"\nMax Absolute Gradient Difference: {grad_diff:.4e}")
        print(f"Max Relative Gradient Difference: {grad_rel:.4e}")
        
        if loss_diff < 1e-10 and grad_diff < 1e-10:
            print("\nSUCCESS: Python adjoint gradient matches Julia Zygote exactly within machine precision bounds!")
        else:
            print("\nWARNING: Numerical differences detected!")
            
    except KeyError as e:
        print(f"Could not load exact verification data from Julia: {e}")

if __name__ == "__main__":
    run_benchmark()
