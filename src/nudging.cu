#ifdef DA_NUDGING

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "assimilate.h"
#include "lbm.h"
#include "data.h"
#include "config.h"
#include "util/stringutils.hpp"
#include "util/port.hpp"
#include "util/mpi_safe_call.hpp"

namespace port = util::port;
#ifdef PORT_CUDA
using backend = port::cuda;
#else
using backend = port::openmp;
#endif

// config of nudging
static constexpr int xyprune = config::da_xyprune;
static constexpr bool da_w_itp = true; // data assimilation with spatial-interpolation
static constexpr auto timeprune = config::daprune;
static constexpr bool skip_east = (config::da_quadra % 2 == 1);
static constexpr bool skip_north = (config::da_quadra >= 2);
static constexpr real alpha = config::da_nud_rate;

void DataAssimilator::init_nudging() {
    constexpr int nxy = config::nx * config::ny;
    uo.reallocate(nxy);
    vo.reallocate(nxy);
    ro.reallocate(nxy);
    uo_host.resize(nxy);
    vo_host.resize(nxy);
    ro_host.resize(nxy);
}

void DataAssimilator::assimilate_nudging_uv(data& dat, const int t) {
    if(config::verbose >= 1)  { std::cout << __PRETTY_FUNCTION__ << ": t=" << t << std::endl; }
    auto step = t / config::iiter;
    if(step % timeprune != 0) { std::cout << __PRETTY_FUNCTION__ << ": t=" << t << ": skip" << std::endl; return; }
    std::cout << __PRETTY_FUNCTION__ << ": nudging_rate=" << alpha << std::endl;

    auto* f = dat.d().f.data();
    auto* rho = dat.d().r.data();
    auto* du_host = this->uo_host.data();
    auto* dv_host = this->vo_host.data();
    auto* dr_host = this->ro_host.data();
    auto* du = this->uo.ptr();
    auto* dv = this->vo.ptr();
    auto* dr = this->ro.ptr();

    { /// load observation u
        auto fname = "io/observed/ens0000/u_step" + util::to_string_aligned(step, 10) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
        file.read(reinterpret_cast<char*>(du_host), sizeof(real) * config::nx * config::ny);
        CUDA_SAFE_CALL(cudaMemcpy(du, du_host, sizeof(real) * config::nx * config::ny, cudaMemcpyHostToDevice));
    }
    { /// load observation v
        auto fname = "io/observed/ens0000/v_step" + util::to_string_aligned(step, 10) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
        file.read(reinterpret_cast<char*>(dv_host), sizeof(real) * config::nx * config::ny);
        CUDA_SAFE_CALL(cudaMemcpy(dv, dv_host, sizeof(real) * config::nx * config::ny, cudaMemcpyHostToDevice));
    }
    { /// load observation rho
        #ifdef DA_OBS_RUV
        auto fname = "io/observed/ens0000/rho_step" + util::to_string_aligned(step, 10) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        runtime_assert(file.is_open(), "IOError: could not open file: " + fname);
        file.read(reinterpret_cast<char*>(dr_host), sizeof(real) * config::nx * config::ny);
        CUDA_SAFE_CALL(cudaMemcpy(dr, dr_host, sizeof(real) * config::nx * config::ny, cudaMemcpyHostToDevice));
        #endif
    }

    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            if(
             (da_w_itp or (i%xyprune==0 && j%xyprune==0))
             &&
             (!skip_east or i < config::nx/2)
             &&
             (!skip_north or j < config::ny/2)
            ) {
                const auto ij = config::ij(i, j);
                real u_obse = 0, v_obse = 0, rho_obse = 0;
                real alpha_alpha = alpha;
                if(xyprune == 1 or !da_w_itp) {
                    u_obse = du[ij];
                    v_obse = dv[ij];
                    #ifdef DA_OBS_RUV
                    rho_obse = dr[ij];
                    #endif
                } else {
                    // 観測xypruneして線形補間
                    const auto iw = (i / xyprune) * xyprune;
                    const auto ie = iw + xyprune;
                    const auto js = (j / xyprune) * xyprune;
                    const auto jn = js + xyprune;
                    const auto ij0 = config::ij_periodic(iw, js);
                    const auto ije = config::ij_periodic(ie, js);
                    const auto ijn = config::ij_periodic(iw, jn);
                    const auto ijen = config::ij_periodic(ie, jn);
                    const auto ir = (i - iw) / real(xyprune);
                    const auto jr = (j - js) / real(xyprune);
                    u_obse = (1.-ir)*(1.-jr)*du[ij0]
                             + ir*(1.-jr)*du[ije]
                             + (1.-ir)*jr*du[ijn]
                             + ir*jr*du[ijen];
                    v_obse = (1.-ir)*(1.-jr)*dv[ij0]
                             + ir*(1.-jr)*dv[ije]
                             + (1.-ir)*jr*dv[ijn]
                             + ir*jr*dv[ijen];
                    #ifdef DA_OBS_RUV
                    rho_obse = (1.-ir)*(1.-jr)*dr[ij0]
                             + ir*(1.-jr)*dr[ije]
                             + (1.-ir)*jr*dr[ijn]
                             + ir*jr*dr[ijen];
                    #endif
                    //alpha_alpha = alpha * sqrt(1e-6 + (0.5-ir)*(0.5-ir) + (0.5-jr)*(0.5-jr)) / 0.5*1.412;
                    //alpha_alpha /= xyprune;
                }

                #ifndef DA_OBS_RUV
                const auto rho_calc = rho[ij];
                rho_obse = rho_calc;
                #endif

                for(int q=0; q<config::Q; q++) {
                    const auto ijq = config::ijq(ij, q);
                    const auto f_calc = f[ijq];
                    const auto f_obse = feq(rho_obse, u_obse, v_obse, q);
                    f[ijq] = alpha_alpha*f_obse + (1-alpha_alpha)*f_calc;
                }
            } // if da_w_itp...
        }
    );

}

#if 0
void DataAssimilator::assimilate_nudging_lbm(data& dat, const int t) {
    if(config::verbose >= 1)  { std::cout << __PRETTY_FUNCTION__ << ": t=" << t << std::endl; }
    auto step = t / config::iiter;
    if(step % timeprune != 0) { std::cout << __PRETTY_FUNCTION__ << ": t=" << t << ": skip" << std::endl; return; }

    auto* f = dat.d().f.data();
    auto* rho = dat.d().r.data();
    auto* u = dat.d().r.data();
    auto* v = dat.d().r.data();
    auto* df = this->obse.data();


    { /// load observation f
        auto fname = "io/observed/0/f_" + std::to_string(step) + ".dat";
        auto file = std::ifstream(fname, std::ios::binary);
        auto size = config::nx * config::ny * config::Q;
        runtime_assert(file.read(reinterpret_cast<char*>(df), sizeof(real) * size), "IOError: could not open file: " + fname);
    }

    port::pfor3d<backend>(
        port::thread3d(config::nx, config::ny, 1),
        [=] __host__ __device__ (port::thread3d, int i, int j, int) {
            if(
             (da_w_itp or (i%xyprune==0 && j%xyprune==0))
             &&
             (!skip_east or i < config::nx/2)
             &&
             (!skip_north or j < config::ny/2)
            ) {
                const auto ij = config::ij(i, j);
                real f_obse[config::Q];
                real alpha_alpha = alpha;
                if(xyprune == 1 or !da_w_itp) {
                    for(int q=0; q<config::Q; q++) {
                        f_obse[q] = df[config::ijq(ij, q)];
                    }
                } else {
                    // 観測xypruneして線形補間
                    const auto iw = (i / xyprune) * xyprune;
                    const auto ie = iw + xyprune;
                    const auto js = (j / xyprune) * xyprune;
                    const auto jn = js + xyprune;
                    const auto ir = (i - iw) / real(xyprune);
                    const auto jr = (j - js) / real(xyprune);
                    for(int q=0; q<config::Q; q++) {
                        f_obse[q] = (1.-ir)*(1.-jr)*df[config::ijq(iw, js, q)]
                                    + ir*(1.-jr)   *df[config::ijq(ie, js, q)]
                                    + (1.-ir)*jr   *df[config::ijq(iw, jn, q)]
                                    + ir*jr        *df[config::ijq(ie, jn, q)];
                        //alpha_alpha = alpha * sqrt(1e-6 + (0.5-ir)*(0.5-ir) + (0.5-jr)*(0.5-jr)) / 0.5*1.412;
                        //alpha_alpha /= xyprune;
                    }
                }

                // nudging update
                real fc[config::Q];
                for(int q=0; q<config::Q; q++) {
                    const auto ijq = config::ijq(ij, q);
                    fc[q] = alpha_alpha*f_obse[q] + (1-alpha_alpha)*f[ijq];
                    f[ijq] = fc[q];
                }
                const auto rhoc = rf(fc);
                const auto uc = uf(fc, rhoc);
                const auto vc = vf(fc, rhoc);
                rho[ij] = rhoc;
                u[ij] = uc;
                v[ij] = vc;
            } // if da_w_itp...
        }
    );

}
#endif

#endif
