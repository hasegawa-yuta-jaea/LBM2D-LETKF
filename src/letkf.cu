#ifdef DA_LETKF
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <sys/stat.h> // mkdir

#include "assimilate.h"
#include "lbm.h"
#include "data.h"
#include "config.h"
#include "util/stringutils.hpp"
#include "util/mpi_safe_call.hpp"
#include "util/cuda_safe_call.hpp"
#include "util/invoker.hpp"
#include "util/runtime_error.hpp"

#include "util/port.hpp"

#ifdef DEBUG
#define DEBUG_PRINT() if (mpi.rank() == 0) { std::cout << "[D] " << __PRETTY_FUNCTION__ << ": " << __LINE__ << std::endl; }
#else
#define DEBUG_PRINT() {}
#endif

// config of letkf
static constexpr auto xyprune = config::da_xyprune;
static_assert(config::nx % xyprune == 0);
static constexpr auto timeprune = config::daprune;
static constexpr real beta = LETKF_COVINF; // covariance inflation factor
static constexpr bool need_save_mat = false;

static constexpr int n_batch0 = config::nx * config::ny;
int n_batch_sp(const util::mpi& mpi) { runtime_assert(n_batch0 % mpi.size() == 0, "batch vs. mpi undivisable"); return n_batch0 / mpi.size(); }
int i_batch_sp_begin(const util::mpi& mpi) { return n_batch_sp(mpi) * mpi.rank(); }

#if   defined(LETKF_X_RUV)
static constexpr int ns_per_grid = 3;  // rho, u, v
#else // LETKF_X_F
static constexpr int ns_per_grid = config::Q;
#endif

#if   defined(DA_OBS_F)
static constexpr int no_per_grid = config::Q;
#else
static constexpr int no_per_grid = 3; // rho, u, v
#endif

static constexpr int n_stt = ns_per_grid;

#ifndef LETKF_RLOC_LEN
static constexpr real d_local = xyprune - 1;
static constexpr int n_obs_local = xyprune == 1 ? /* pointwise: */ 0 : /* r-localization c=p-1: */ 2;
static constexpr int obs_offset  = xyprune == 1 ? /* pointwise: */ 0 : /* r-localization c=p-1: */ 1;
static constexpr int n_obs_x     = xyprune == 1 ? /* pointwise: */ 1 : /* r-localization c=p-1: */ 4;
#else
static constexpr real d_local = LETKF_RLOC_LEN * xyprune;
static constexpr int obs_offset = 0;
static constexpr int n_obs_local = LETKF_RLOC_LEN * 2;
static constexpr int n_obs_x = n_obs_local * 2 + 1;
#endif
static constexpr int n_obs = n_obs_x * n_obs_x * no_per_grid;


void DataAssimilator::init_letkf(const util::mpi& mpi) {
    xk.reallocate(n_batch0 * n_stt);
    xk_host.reallocate(n_batch0 * n_stt);
    yk.reallocate(n_batch0 * n_obs);

    #if defined(DA_OBS_F)
    {
        constexpr int nxyq = config::nx * config::ny * config::Q;
        fo.reallocate(nxyq);
        fo_host.resize(nxyq);
    }
    #else  // DA_OBS_RUV
    {
        constexpr int nxy = config::nx * config::ny;
        uo.reallocate(nxy);
        vo.reallocate(nxy);
        ro.reallocate(nxy);
        uo_host.resize(nxy);
        vo_host.resize(nxy);
        ro_host.resize(nxy);
    }
    #endif

    const int n_batch = n_batch_sp(mpi);
    const int i_batch0 = i_batch_sp_begin(mpi);
    runtime_assert(i_batch0 % config::nx == 0, "batch size per mpi should be divisable by nx");
    solver.init_geo(mpi.size(), n_stt, n_obs, n_batch);
    solver.set_rR(dim3(1, 1, n_batch), dim3(n_obs_x, n_obs_x),
        [=] __device__ (letkf_real* e, int n_rows, int stride) {
            // obs id
            const int i_obs = threadIdx.x;
            const int j_obs = threadIdx.y;
            const int ij_obs = i_obs + n_obs_x*j_obs;
            // grid offset by i_batch
            const int i_batch  = blockIdx.z;
            const int i0_stt = 0;
            const int j0_stt = i_batch0 / config::nx;
            // grid id
            const int i_stt = i0_stt + i_batch % config::nx;
            const int j_stt = j0_stt + i_batch / config::nx;

            // obs error cov
            const int k = ij_obs + ij_obs*n_rows + i_batch*stride;
            const int kofs = n_obs_x*n_obs_x + (n_obs_x*n_obs_x)*n_rows;
            #if defined(DA_OBS_F)
            {
                for(int q=0; q<config::Q; q++) {
                    const auto efo = config::obs_error_f * feq(config::rho_ref, 0, 0, q);
                    e[k + q*kofs] = powf(efo, real(-2));
                }
                // R-localization
                if(xyprune > 1) {
                    constexpr real cutoff = d_local + 1e-6;
                    const int io_stt = (int(i_stt/xyprune) + i_obs - (n_obs_local-obs_offset))*xyprune;
                    const int jo_stt = (int(j_stt/xyprune) + j_obs - (n_obs_local-obs_offset))*xyprune;
                    const real di = i_stt - io_stt;
                    const real dj = j_stt - jo_stt;
                    const real d = sqrtf(di*di + dj*dj) / cutoff;
                    const real gaspari_cohn =
                        (d <= 0) ? 1:
                        (d <= 1) ? -d*d*d*d*d/4. + d*d*d*d/2. + d*d*d*5./8. - d*d*5./3. + 1:
                        (d <= 2) ? d*d*d*d*d/12. - d*d*d*d/2. + d*d*d*5./8. + d*d*5./3. - d*5 + 4 - 2./3./d:
                        0;
                    for(int q=0; q<config::Q; q++) {
                        const auto efo = config::obs_error_f * feq(config::rho_ref, 0, 0, q);
                        e[k + q*kofs] *= gaspari_cohn;
                    }
                }
            }
            #elif defined(DA_OBS_RUV)
            {
                e[k] = powf(config::obs_error_u, real(-2)); // u
                e[k + kofs] = powf(config::obs_error_u, real(-2)); // v
                e[k + 2*kofs] = powf(config::obs_error_rho, real(-2)); // rho
                // R-localization
                if(xyprune > 1) {
                    constexpr real cutoff = d_local + 1e-6;
                    const int io_stt = (int(i_stt/xyprune) + i_obs - (n_obs_local-obs_offset))*xyprune;
                    const int jo_stt = (int(j_stt/xyprune) + j_obs - (n_obs_local-obs_offset))*xyprune;
                    const real di = i_stt - io_stt;
                    const real dj = j_stt - jo_stt;
                    const real d = sqrtf(di*di + dj*dj) / cutoff;
                    const real gaspari_cohn =
                        (d <= 0) ? 1:
                        (d <= 1) ? -d*d*d*d*d/4. + d*d*d*d/2. + d*d*d*5./8. - d*d*5./3. + 1:
                        (d <= 2) ? d*d*d*d*d/12. - d*d*d*d/2. + d*d*d*5./8. + d*d*5./3. - d*5 + 4 - 2./3./d:
                        0;
                    e[k] *= gaspari_cohn;
                    e[k + kofs] *= gaspari_cohn;
                    e[k + 2*kofs] *= gaspari_cohn;
                }
            }
            #else
            #error da_obs mode not implemented
            #endif


        });
    solver.set_covariance_inflation(beta);
    if(mpi.rank() == 0) { solver.inspect(); }
    mpi.barrier();
}

void DataAssimilator::assimilate_letkf(data& dat, const util::mpi& mpi, const int t) {
    timer.stop_and_ignore_latter();
    auto step = t / config::iiter;
    if(step % timeprune != 0) { return; }
    if(config::verbose >= 1 && mpi.rank() == 0)  { std::cout << __PRETTY_FUNCTION__ << ": t=" << t << std::endl; }

    const auto* f = dat.d().f.data();
    const auto* rho = dat.d().r.data();
    const auto* u = dat.d().u.data();
    const auto* v = dat.d().v.data();
    MPI_Request mpi_requests[4]; // load_obs[3]+yk[1], xk[1]

    DEBUG_PRINT();
    timer.transit("pack_xk");
    auto* xk = this->xk.ptr();
    #if   defined(LETKF_X_RUV)
    {
        util::invoke_device<<< config::ny, config::nx >>>(
            [=] __device__ () {
                const int i_batch = threadIdx.x + blockDim.x * blockIdx.x;
                real fc[config::Q];
                for(int q=0; q<config::Q; q++) {
                    const int i_lbm = config::ijq(i_batch, q);
                    fc[q] = f[i_lbm];
                }
                struct state_vec { real rc, uc, vc; };
                auto rc = rf(fc);
                auto uc = uf(fc, rc);
                auto vc = vf(fc, rc);
                state_vec v = { rc, uc, vc };
                reinterpret_cast<state_vec*>(xk)[i_batch] = v;
            }
        );
    }
    #else // LETKF_X_F
    {
        util::invoke_device<<< dim3(config::ny, config::Q), config::nx >>>(
            [=] __device__ () {
                const int i = threadIdx.x + blockDim.x * blockIdx.x;
                const int q = blockIdx.y;
                const int i_lbm = config::ijq(i, q);
                const int i_mat = q + config::Q * i;
                xk[i_mat] = f[i_lbm];
            }
        );
    }
    #endif
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(xk_host.ptr(), xk, xk_host.size()*sizeof(letkf_real), cudaMemcpyDeviceToHost));

    DEBUG_PRINT();
    timer.transit("pack_yk");
    auto* yk = this->yk.ptr();
    util::invoke_device<<< dim3(config::ny, config::nx), dim3(n_obs_x, n_obs_x) >>>(
        [=] __device__ () {
            // indices of the rows in the batch
            const int ir = threadIdx.x;
            const int jr = threadIdx.y;
            const int r = ir + jr * n_obs_x;
            // index of the batch
            const int it = blockIdx.x;
            const int jt = blockIdx.y;
            const int bt = it + config::nx * jt;
            // indices of the grid (offset by batch)
            const int ig = it;
            const int jg = jt;
            // indices of the grid
            const int io = (int(ig/xyprune) + ir - (n_obs_local-obs_offset))*xyprune;
            const int jo = (int(jg/xyprune) + jr - (n_obs_local-obs_offset))*xyprune;
            const int di = abs(ig - io);
            const int dj = abs(jg - jo);

            const int iop = (io + config::nx) % config::nx;
            const int jop = (jo + config::ny) % config::ny;

            #if defined(DA_OBS_F)
            {
                for(int q=0; q<config::Q; q++) {
                    const auto i = config::ijq(iop, jop, q);
                    const auto k = r + n_obs*bt;
                    constexpr auto kofs = n_obs_x * n_obs_x;
                    yk[k + q*kofs] = f[i];
                }
            }
            #elif defined(DA_OBS_RUV)
            {
                // pack u, v
                real fc[config::Q];
                for(int q=0; q<config::Q; q++) {
                    const auto i = config::ijq(iop, jop, q);
                    fc[q] = f[i];
                }
                auto rc = rf(fc);
                auto uc = uf(fc, rc);
                auto vc = vf(fc, rc);
                if(di > n_obs_local*xyprune || dj > n_obs_local*xyprune) { uc=0; vc=0; rc=0; }
                const auto k = r + n_obs*bt;
                yk[k] = uc;
                yk[k + n_obs_x*n_obs_x] = vc;
                yk[k + 2*n_obs_x*n_obs_x] = rc;
            }
            #endif
        }
    );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    DEBUG_PRINT();
    timer.transit("mpi_yk");
    solver.mpi_ialltoall_yk(mpi.comm(), &mpi_requests[3], yk);

    // comm overlap: load_obs@master&MPI_Ibcast(obse), MPI_Alltoall(yk)
    DEBUG_PRINT();
    timer.transit("load_obs");
    MPI_Datatype MPI_REAL_ = util::mpi::MPItypename<real>::name();
    #if defined(DA_OBS_F)
    {
        if(mpi.rank() == 0) { /// load observation f
            const auto fname = "io/observed/ens0000/f_step" + util::to_string_aligned(step, 10) + ".dat";
            auto file = std::ifstream(fname, std::ios::binary);
            runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
            file.read(reinterpret_cast<char*>(fo_host.data()), sizeof(real) * config::nx * config::ny * config::Q);
            CUDA_SAFE_CALL(cudaMemcpy(fo.ptr(), fo_host.data(), sizeof(real) * config::nx * config::ny * config::Q, cudaMemcpyHostToDevice));
        }
        MPI_Bcast(fo.ptr(), /* cnt= */config::nx*config::ny*config::Q, MPI_REAL_, /* root= */0, mpi.comm());
    }
    #elif defined(DA_OBS_RUV)
    {
        if(mpi.rank() == 0) { /// load observation r
            const auto fname = "io/observed/ens0000/rho_step" + util::to_string_aligned(step, 10) + ".dat";
            auto file = std::ifstream(fname, std::ios::binary);
            runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
            file.read(reinterpret_cast<char*>(ro_host.data()), sizeof(real) * config::nx * config::ny);
            CUDA_SAFE_CALL(cudaMemcpy(ro.ptr(), ro_host.data(), sizeof(real) * config::nx * config::ny, cudaMemcpyHostToDevice));
        }
        MPI_Ibcast(ro.ptr(), /* cnt= */config::nx*config::ny, MPI_REAL_, /* root= */0, mpi.comm(), &mpi_requests[0]);
        if(mpi.rank() == 0) { /// load observation u
            const auto fname = "io/observed/ens0000/u_step" + util::to_string_aligned(step, 10) + ".dat";
            auto file = std::ifstream(fname, std::ios::binary);
            runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
            file.read(reinterpret_cast<char*>(uo_host.data()), sizeof(real) * config::nx * config::ny);
            CUDA_SAFE_CALL(cudaMemcpy(uo.ptr(), uo_host.data(), sizeof(real) * config::nx * config::ny, cudaMemcpyHostToDevice));
        }
        MPI_Ibcast(uo.ptr(), /* cnt= */config::nx*config::ny, MPI_REAL_, /* root= */0, mpi.comm(), &mpi_requests[1]);
        if(mpi.rank() == 0) { /// load observation v
            const auto fname = "io/observed/ens0000/v_step" + util::to_string_aligned(step, 10) + ".dat";
            auto file = std::ifstream(fname, std::ios::binary);
            runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
            file.read(reinterpret_cast<char*>(vo_host.data()), sizeof(real) * config::nx * config::ny);
            CUDA_SAFE_CALL(cudaMemcpy(vo.ptr(), vo_host.data(), sizeof(real) * config::nx * config::ny, cudaMemcpyHostToDevice));
        }
        MPI_Ibcast(vo.ptr(), /* cnt= */config::nx*config::ny, MPI_REAL_, /* root= */0, mpi.comm(), &mpi_requests[2]);
        MPI_Waitall(4, mpi_requests, MPI_STATUSES_IGNORE);
    }
    #endif  // ifdef DA_OBS_*
    mpi.barrier();

    // finalize yk after comm
    DEBUG_PRINT();
    timer.transit("pack_yk");
    solver.reorder_yk_after_ialltoall();

    DEBUG_PRINT();
    timer.transit("set_yo");
    runtime_assert(config::ny % mpi.size() == 0, "grid number vs. mpi is undivisible");
    const auto* uo = this->uo.ptr();
    const auto* vo = this->vo.ptr();
    const auto* ro = this->ro.ptr();
    const auto* fo = this->fo.ptr();
    const int yo_n_batch_y = config::ny / mpi.size();
    const int yo_j_batch_offset = i_batch_sp_begin(mpi) / config::nx;
    solver.set_yo(0, dim3(config::nx, yo_n_batch_y), dim3(n_obs_x, n_obs_x),
        [=] __device__ (letkf_real* yo, size_t stride) {
            // indices of the rows in the batch
            const int ir = threadIdx.x;
            const int jr = threadIdx.y;
            const int r = ir + jr * n_obs_x;
            // index of the batch
            const int it = blockIdx.x;
            const int jt = blockIdx.y;
            const int bt = it + config::nx * jt;
            // indices of the grid (offset by batch)
            const int ig = it;
            const int jg = jt + yo_j_batch_offset;
            // indices of the grid
            const int io = (int(ig/xyprune) + ir - (n_obs_local-obs_offset))*xyprune;
            const int jo = (int(jg/xyprune) + jr - (n_obs_local-obs_offset))*xyprune;
            const int di = abs(ig - io);
            const int dj = abs(jg - jo);
            const int iop = (io + config::nx) % config::nx;
            const int jop = (jo + config::ny) % config::ny;

            #if defined(DA_OBS_F)
            {
                for(int q=0; q<config::Q; q++) {
                    const auto ijq = config::ijq(iop, jop, q);
                    const auto k = r + n_obs*bt;
                    constexpr auto kofs = n_obs_x * n_obs_x;
                    auto fc = fo[ijq];
                    if(di > n_obs_local*xyprune || dj > n_obs_local*xyprune) { fc=0; }
                    yo[k + q*kofs] = fc;
                }
            }
            #elif defined(DA_OBS_RUV)
            {
                // pack u, v
                const int i = config::ij(iop, jop);
                const int k = r + n_obs*bt;
                auto uc = uo[i];
                auto vc = vo[i];
                auto rc = ro[i];
                if(di > n_obs_local*xyprune || dj > n_obs_local*xyprune) { uc=0; vc=0; rc=0; }
                yo[k] = uc;
                yo[k + n_obs_x*n_obs_x] = vc;
                yo[k + 2*n_obs_x*n_obs_x] = rc;
            }
            #endif  // ifdef DA_OBS_*
        } // [=] __device__
    ); // solver.ser_yo

    DEBUG_PRINT();
    timer.transit("mpi_xk");
    solver.mpi_ialltoall_xk_host(mpi.comm(), &mpi_requests[0], xk_host.ptr());

    { /// solver part with overlapped communication of mpi_ialltoall_xk
        DEBUG_PRINT();
        timer.transit("solve_comm_ovlp");
        std::string mat_save_prefix =
            need_save_mat
            ? config::prefix + "/letkf/mat/ens" + std::to_string(mpi.rank())
            : "";
        ::mkdir(mat_save_prefix.c_str(), 0755);
        solver.solve_evd(0, mat_save_prefix, t);
    }

    #ifndef LETKF_NO_MPI_BARRIER
    {
        timer.transit("mpi_barrier");
        mpi.barrier();
    }
    #endif
    DEBUG_PRINT();
    timer.transit("mpi_xk");
    MPI_Wait(&mpi_requests[0], MPI_STATUSES_IGNORE);
    DEBUG_PRINT();
    timer.transit("pack_xk");
    solver.copy_xk_host_to_device();
    solver.reorder_xk_after_ialltoall();

    { /// non overlapped solver part
        DEBUG_PRINT();
        timer.transit("solve");
        std::string mat_save_prefix =
            need_save_mat
            ? config::prefix + "/letkf/mat/ens" + std::to_string(mpi.rank())
            : "";
        ::mkdir(mat_save_prefix.c_str(), 0755);
        solver.solve_xa(0, mat_save_prefix, t);
    }

    #ifndef LETKF_NO_MPI_BARRIER
    {
        timer.transit("mpi_barrier");
        mpi.barrier();
    }
    #endif
    DEBUG_PRINT();
    timer.transit("mpi_xsol");
    solver.mpi_alltoall_Xsol(mpi.comm());

    // update f by LETKF sol
    DEBUG_PRINT();
    timer.transit("update_xk");
    auto* fn = dat.dn().f.data();
    #if   defined(LETKF_X_RUV)
    solver.update_xk(
        0, // id of ens (dummy)
        0, // i_batch_0
        config::ny, config::nx,
        [=] __device__ (const letkf_real* Xk, size_t stride) {
            const auto i_batch = threadIdx.x + blockDim.x * blockIdx.x;

            struct state_vec { real rc, uc, vc; };
            state_vec v = reinterpret_cast<const state_vec*>(Xk)[i_batch];

            // compute feq, fneq; assimilate feq only
            real fc[config::Q];
            for(int q=0; q<config::Q; q++) {
                const auto idx_lbm = config::ijq(i_batch, q);
                fc[q] = f[idx_lbm];
            }
            auto rc = rf(fc);
            auto uc = uf(fc, rc);
            auto vc = vf(fc, rc);

            for(int q=0; q<config::Q; q++) {
                const auto feq_assim = feq(v.rc, v.uc, v.vc, q);
                const auto fneq = fc[q] - feq(rc, uc, vc, q);
                const auto idx_lbm = config::ijq(i_batch, q);
                fn[idx_lbm] = feq_assim + fneq;
            }
        }
    );

    #else // LETKF_X_F
    solver.update_xk(
        0, // id of ens (dummy)
        0, // i_batch_0
        dim3(config::ny, config::Q), config::nx,
        [=] __device__ (const letkf_real* Xk, size_t stride) {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            const auto q = blockIdx.y;
            const auto idx_lbm = config::ijq(i, q);
            const auto idx_xk = q + n_stt*i;
            fn[idx_lbm] = Xk[idx_xk];
        }
    );

    #endif // ifdef LETKF_X_*
    DEBUG_PRINT();

    // update macro by fn
    timer.transit("lbm2euler");
    auto* rn = dat.dn().r.data();
    auto* un = dat.dn().u.data();
    auto* vn = dat.dn().v.data();
    util::invoke_device<<<config::nx, config::ny>>>(
        [=] __device__ () {
            const auto i = threadIdx.x + blockDim.x * blockIdx.x;
            // lbm
            real fc[config::Q];
            #pragma unroll
            for(int q=0; q<config::Q; q++) {
                fc[q] = fn[config::ijq(i, q)];
            }
            const real rnc = rf(fc);
            const real unc = uf(fc, rnc);
            const real vnc = vf(fc, rnc);
            rn[i] = rnc;
            un[i] = unc;
        }
    );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    dat.swap();

    timer.stop_and_ignore_latter();
}

#endif // ifdef DA_LETKF
