from ffsim_hubbard_vqe2 import _get_gso_indices
import ffsim
import numpy as np

n_sites = 6
n_particles = (3, 3)

dim_spatial = ffsim.dim(n_sites, n_particles)
gso_indices = _get_gso_indices(dim_spatial, n_sites, n_particles)

gso_norb = 2 * n_sites
gso_nelec = (sum(n_particles), 0)
dim_gso = ffsim.dim(gso_norb, gso_nelec)

indices = np.arange(dim_gso, dtype=int)
strings = ffsim.addresses_to_strings(indices, gso_norb, gso_nelec)
strings = np.array(strings, dtype=int)

# HF index in spatial is 0 (usually)
hf_idx_spatial = 0
hf_gso_idx = gso_indices[hf_idx_spatial]
hf_string = strings[hf_gso_idx]

print(f"Spatial HF Index: {hf_idx_spatial}")
print(f"Mapped GSO Index: {hf_gso_idx}")
print(f"Mapped GSO String: {bin(hf_string)}")

# Expected: 3 alpha (0,1,2), 3 beta (0,1,2) -> GSO (0,1,2, 6,7,8)
# String should have bits 0,1,2 and 6,7,8 set.
expected_mask = (1<<0)|(1<<1)|(1<<2) | (1<<6)|(1<<7)|(1<<8)
print(f"Expected String:   {bin(expected_mask)}")

if hf_string == expected_mask:
    print("MAPPING MATCHES HF STATE!")
else:
    print("MAPPING MISMATCH!")
