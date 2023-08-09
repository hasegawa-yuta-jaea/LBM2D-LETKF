# About LBM2D-LETKF

<img width=480 src=doc/header.png />

`LBM2D-LETKF` is a microbenchmark code for ensemble data assimilation of turbulent flow using the 2D lattice Boltzmann method (LBM) and local ensemble transform Kalman filter (LETKF).
Computation is fully implemented in NVIDIA GPU by using CUDA, cuBLAS, cuSOLVER.
The problem size of the test is typically 256 x 256 grid points with {4,16,64} ensembles, where the required number of GPUs is equal to the ensemble size (4, 16, or 64 GPUs).

This microbenchmark code has been developed to test the capability of ensemble data assimilation to `CityLBM`.
`CityLBM` is an application for real-time high-resolution simulation of urban wind flow and plume dispersion. 
See [our publication [Onodera2021]](https://doi.org/10.1007/s10546-020-00594-x) for detail.

Differences between `CityLBM` and this benchmark are summarized as follows:

| | `CityLBM` | `LBM2D-LETKF` |
|:--|:--|:--|
| dimension | 3D | 2D |
| problem | realistic problem of wind flow and plume dispersion in urban areas with high-rise buildings | virtually configured (non-realistic) isotropic turbulence |
| mesh type | locally refined mesh (AMR) | uniform mesh |
| mesh size | > O(10^8) | O(10^4) |
| numerical model for velocity | LBM<br>cumulant collision [[Geier2015]](https://doi.org/10.1016/j.camwa.2015.05.001)<br>CSM-LES [[Kobayashi2005]](https://doi.org/10.1063/1.1874212) | LBM<br>standard BGK collision<br>standard Smagorinsky model |
| numerical model for temperature | Boussinesq approximation with finite difference discretization | *None* |
| numerical model for plume dispersion | passible scalar with finite volume discretization | *None* |
| implement of LEKTF | *not yet* | yes |

Besides, the LETKF is a well-known method of ensemble data assimilation with high-performance computing in numerical weather prediction (NWP) community; for details, cf. e.g. [[Hunt2007]](https://doi.org/10.1016/j.physd.2006.11.008), [[Miyoshi2016]](https://doi.org/10.1002/2014GL060863), [[Yashiro2020]](https://doi.org/10.1109/SC41405.2020.00005).
    
## Usage

### Preparation

Firstly, the following external libraries are required:

- GCC >= 7.4.0
- CUDA >= 11.0
- CUDA-aware MPI (e.g. module `ompi-cuda` at Wisteria-A at ITC UTokyo, `mpt/2.23-ga` at HPE SGI8600 at JAEA)
- HDF5 (ver 1.8.2 or 1.12.0 may be available)
- Boost C++

**Optional:** As eigenvalue decomposition solver for LETKF, while `cuSOLVER` is used as default, [`EigenG-Batched`](https://www.r-ccs.riken.jp/labs/lpnctrt/projects/eigengbatched/index.html), an open-source code released from RIKEN, is also available.
(Due to a performance reason, `EigenG-Batched` is highly recommended for the larger ensemble size M>32)

To use `EigenG-Batched`, execute
```
./config/enable_EigenG-Batched.sh
```

### Compile

Select one of the test modes from below:

| Test Mode | Description |
| -- | -- |
| `NATURE` | Compute a ground-truth |
| `OBSERVE` | Create observation data from nature run |
| `LYAPNOV` | Test Lyapunov exponent of the system *(TBD)* |
| `DA_NUDGING` | Compute data assimilation experiment using nudging method *(TBD)* |
| `DA_LETKF` | Compute data assimilation experiment using LETKF |
| `DA_DUMMY` | *dummy* |

For example, if you want to compute a ground-truth,
```bash
make clean
TEST=NATURE make
```
After compilation is succeeded, the executable named `run/a.out` is generated.

Or if you want to check the compilation of every mode, run
```bash
make test
```

### Run

#### (1) Short validation test

To check the validity of the data assimilation, one may test the Observing System Simulation Experiment (OSSE) of LBM2D-LETKF.
The following steps do the OSSE calculation:

1. Compute nature run and create observational data:
   ```
   make clean
   make TEST=OBSERVE
   mpiexec <mpiexec_options...> -np 1 run/a.out
   ```
   \*\* note: `mpiexec_options` depend on environments. For details, please refer to the example job scripts, [`run/wistimer`](run/wistimer) (for Wisteria-A) or [`run/jstimer`](run/jstimer) (for HPE SGI8600).
2. Compute data assimilation:
   ```
   make clean
   make TEST=DA_LETKF
   mpiexec <mpiexec_options...> -np <ensemble_size> run/a.out
   ```
   \*\* note: `ensemble_size` is, e.g. 4, 16, or 64
3. Visualize the result; this will generate the snapshots of the contour of the vorticity field in nature run, observation, and LETKF:
   ```
   ./postprocess/plot-vorticity.py io/*/ens*/
   ```
   
After 3., you may find the snapshots named `io/*/0/vor_1990.png` (an example is shown below).
The result of the LETKF is expected to be similar to the Nature run; otherwise, the validation test has failed. 

| Nature run | Observation | LETKF (expected) | Non-DA or invalid LETKF |
|:--|:--|:--|:--|
| <img width=160 src=doc/nature_vor_step0000002000.png> | <img width=160 src=doc/observed_vor_step0000002000.png> | <img width=160 src=doc/letkf_vor_step0000002000.png> | <img width=160 src=doc/noda_vor_step0000002000.png> | 

#### (2) performance evaluation

*OUT OF DATE. The below result is with 128 x 128 mesh.*

The validation test also measures the performance, hence additional computation is not needed.
You may find the performance result in `io/elapsed_time_rank*.csv` and `io/letkf_elapsed_time_rank*.csv`.
The result is, for instance,
- `io/elapsed_time_rank0.csv`
  ```
  #tag,sec
  DA,1.68128
  _sync.DA,0.0289409
  _sync.forecast,0.0262086
  _sync.output,0.01511
  forecast,0.55471
  output,1.39295
  ```
  Here, the elapsed time \[sec\] is measured over 100 data assimilation cycles (10000 LBM steps). Each tag means:
  * `DA`: elapsed time of LETKF. Its breakdown is stored in `io/letkf_elapsed_time_rank0.csv`
  * `forecast`: elapsed time of LBM.
  * `output`: elapsed time of file output for validation test.
  * `_sync.*`: overhead. Maybe ignored.

- `io/letkf_elapsed_time_rank0.csv`
  ```
  #tag,sec
  _ignored,2.00556
  lbm2euler,0.00115395
  load_obs,0.270094
  mpi_barrier,0.17283
  mpi_xk,0.0161288
  mpi_xsol,0.190429
  mpi_yk,0.064585
  pack_xk,0.0215552
  pack_yk,0.00485086
  set_yo,0.0119514
  solve,0.114645
  solve_comm_ovlp,0.809463
  update_xk,0.00277638
  ```
  These tags are roughly categorized into the following types:
  * Computation: `lbm2euler`, `set_yo`, `solve`, `solve_comm_ovlp`, `update_xk`
  * File read: `load_obs`
  * Communication: `mpi_xk`, `mpi_xsol`, `mpi_yk`.<br>
    **Note:** Communication is partially overlapped with computation and file read. The elapsed time shown here is of the non-overlapped part.
  * Others:
    - `mpi_barrier`: waiting time to synchronize imbalance of computation.
    - `_ignored`: elapsed times outside the LETKF. Ignored.

Besides, examples of performance evaluations at Wisteria-A can be found in [doc/result_example/aquarius](doc/result_example/aquarius).

## Citation

1. Y. Hasegawa, T. Imamura, T. Ina, N. Onodera, Y. Asahi, and Y. Idomura, "GPU Optimization of Lattice Boltzmann Method with Local Ensemble Transform Kalman Filter," in *2022 IEEE/ACM Workshop on Latest Advances in Scalable Algorithms for Large-Scale Heterogeneous Systems (ScalAH)*, pp. 10-17, 2022. \[doi:[10.1109/ScalAH56622.2022.00007](https://doi.org/10.1109/ScalAH56622.2022.00007), arXiv:[2308.03310](https://arxiv.org/abs/2308.03310), Version of code: [`LBM2D-LETKF:v2.4.4`](https://github.com/hasegawa-yuta-jaea/LBM2D-LETKF/releases/tag/v2.4.4)\]
1. Y. Hasegawa, N. Onodera, Y. Asahi, T. Ina, T. Imamura, and Y. Idomura, "Continuous data assimilation of large eddy simulation by lattice Boltzmann method and local ensemble transform Kalman filter (LBM-LETKF)," *submitted*. \[arXiv:[2308.03972](https://arxiv.org/abs/2308.03972), Version of code: [`LBM2D-LETKF:v3.0.5`](https://github.com/hasegawa-yuta-jaea/LBM2D-LETKF/releases/tag/v3.0.5)\]
