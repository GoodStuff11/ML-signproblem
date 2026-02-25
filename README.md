# ML-signproblem

Run setup with

```bash
git clone -b simplified https://github.com/GoodStuff11/ML-signproblem.git
cd ML-signproblem/experimenting/ed
julia -e 'import Pkg; Pkg.activate("../"); Pkg.instantiate(); Pkg.update()'
julia --project=../ --threads=10 run_lanczos_scan_optimization.jl true
```

