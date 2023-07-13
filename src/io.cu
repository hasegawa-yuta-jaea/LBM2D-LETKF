#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdio> // FILE*
#include <sys/stat.h>

#include "io.h"
#include "data.h"
#include "config.h"
#include "lbm.h"
#include "util/stringutils.hpp"
#include "util/range.hpp"
#include "util/runtime_error.hpp"
#include "util/mpi_wrapper.hpp"

void output(const data& dat, util::mpi mpi, int t) {
    if(t % (config::iiter * config::ioprune) != 0) { return; }
    constexpr auto nx = config::nx, ny = config::ny, Q = config::Q;
    // output
    auto dname = [=](std::string type) { return config::prefix + "/" + type + "/ens" + util::to_string_aligned(dat.ensemble_id); };
    auto fname0 = [=](std::string type, std::string tag) { return dname(type) + "/" + tag; };
    if(t <= 0) { // double check
        ::mkdir(dname("calc").c_str(), 0755);
        if(dat.ensemble_id == 0) {
            ::mkdir(dname("nature").c_str(), 0755);
            ::mkdir(dname("observed").c_str(), 0755);
        }
    }

    auto output_file = [=](std::string name0, const real* ptr, int q=1) {
        std::string fname = name0 + "_step" + util::to_string_aligned(t / config::iiter, 10) + ".dat";
        std::FILE* fp = std::fopen(fname.c_str(), "wb");
        runtime_assert(fp != NULL, "could not open file: " + fname);
        size_t size = config::nx * config::ny * q;
        size_t fwrite_size = std::fwrite(ptr, sizeof(real), size, fp);
        runtime_assert(fwrite_size == size, "IOError: fwrite failed");
        std::fclose(fp);
    };

    if(mpi.rank() == 0 && config::verbose >= 5) {
        std::cout << __PRETTY_FUNCTION__
            << ": t_fout=" << t/config::iiter << ", "
            <<  "t=" << t << std::endl;
    }
    // make output vector
    std::vector<real> vor(nx*ny);
    std::vector<real> r(nx*ny);
    std::vector<real> u(nx*ny);
    std::vector<real> v(nx*ny);
    std::vector<real> f(nx*ny*Q);
    CUDA_SAFE_CALL(cudaMemcpy(r.data(), dat.d().r.data(), sizeof(real)*r.size(), cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(u.data(), dat.d().u.data(), sizeof(real)*u.size(), cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(v.data(), dat.d().v.data(), sizeof(real)*v.size(), cudaMemcpyDefault));
    CUDA_SAFE_CALL(cudaMemcpy(f.data(), dat.d().f.data(), sizeof(real)*f.size(), cudaMemcpyDefault));
    for(int j=0; j<config::ny; j++) {
        #pragma omp parallel for
        for(int i=0; i<config::nx; i++) {
            const auto ij = config::ij(i, j);
            if(config::is_boundary(i, j)) {
                vor.at(ij) = 0;
            } else {
                const int iw = (i-1+config::nx)%config::nx;
                const int ie = (i+1)%config::nx;
                const int js = (j-1+config::ny)%config::ny;
                const int jn = (j+1)%config::ny;
                const real us = u.at(config::ij(i, js));
                const real un = u.at(config::ij(i, jn));
                const real uy = (un - us) / (2*config::dx);
                const real vw = v.at(config::ij(iw, j));
                const real ve = v.at(config::ij(ie, j));
                const real vx = (ve - vw) / (2*config::dx);
                vor.at(ij) = vx - uy;
                #ifdef EXIT_IF_NAN
                {
                    if(std::isnan(vx - uy)) {

                        throw STD_RUNTIME_ERROR("nan value detected");
                    }
                }
                #endif
            }
        }
    }

    #ifdef ENSEMBLE_STAT
    {
        // ensemble
        auto comm = mpi.comm();
        auto iv = real(1) / mpi.size();
        constexpr auto count = config::ny * config::nx;
        auto datatype = util::mpi::MPItypename<real>::name();
        std::vector<real> u_mean(count), v_mean(count);
        std::vector<real> vor_mean(count);
        std::vector<real> vor2_sum(count);
        std::vector<real> vor_std (count);

        if(mpi.size() > 1) {
            /// mean
            //// MPI_Allreduce(): send, recv, count, datatype, op, comm
            MPI_Allreduce(vor.data(), vor_mean.data(), count, datatype, MPI_SUM, comm);
            MPI_Allreduce(u.data(), u_mean.data(), count, datatype, MPI_SUM, comm);
            MPI_Allreduce(v.data(), v_mean.data(), count, datatype, MPI_SUM, comm);
           #pragma omp parallel for
            for(int i=0; i<count; i++) {
                vor_mean[i] *= iv;
                u_mean[i] *= iv;
                v_mean[i] *= iv;
            }

            /// std
            //// mean square
            auto sum2 = [](real a) { return a*a; };
           #pragma omp parallel for
            for(int i=0; i<count; i++) {
                vor2_sum[i] += iv * sum2(vor[i]);
            }
            MPI_Allreduce(vor2_sum.data(), vor_std.data(), count, datatype, MPI_SUM, comm);
            /// mean square --> var --> std
            auto stdev = [](real ms, real m) {
                auto v = ms - m*m;
                auto s = std::sqrt(v);
                return s;
            };
           #pragma omp parallel for
            for(int i=0; i<count; i++) {
                vor_std[i] = stdev(vor_std[i], vor_mean[i]);
            }
        }
        /// output
        if(mpi.rank() == 0) {
            output_file(config::prefix + "/calc/ens_mean_u", u_mean.data());
            output_file(config::prefix + "/calc/ens_mean_v", v_mean.data());
            output_file(config::prefix + "/calc/ens_vor_mean", vor_mean.data());
            output_file(config::prefix + "/calc/ens_vor_std", vor_std.data());
        }
    }
    #endif  // ifdef ENSEMBLE_STAT

    // inspect
    if(mpi.rank() == 0) {
        dat.d().inspect();
        if(config::verbose >= 100) {
            std::cout << "  vor: " << *std::min_element(vor.begin(), vor.end())  << " " << *std::max_element(vor.begin(), vor.end()) << std::endl;
        }
    }


    /// output individual
    #ifndef OBSERVE
    {
        output_file(fname0("calc", "vor"), vor.data());
        output_file(fname0("calc", "rho"), r.data());
        output_file(fname0("calc", "u"), u.data());
        output_file(fname0("calc", "v"), v.data());
        output_file(fname0("calc", "f"), f.data(), config::Q);
    }
    #endif  // ifndef OBSERVE

    /// make observed data
    #ifdef OBSERVE
    {
        if(dat.ensemble_id == 0) {
            const auto nxy = config::nx * config::ny;
            const auto Q = config::Q;

            // observe
            static auto rand_engine = std::mt19937(334 + dat.ensemble_id);
            auto rand_dist = std::normal_distribution<real>(0, 1);
            auto rand = [&]() { return rand_dist(rand_engine); };

            std::vector<real> fo(nxy * Q), ro(nxy), uo(nxy), vo(nxy);

            #ifdef OBSERVE_ERROR_RUV
            {
                for(auto&& ij: util::range(nxy)) {
                    const auto er = config::obs_error_rho;
                    const auto eu = config::obs_error_u;
                    ro.at(ij) = r.at(ij) + er * rand();
                    uo.at(ij) = u.at(ij) + eu * rand();
                    vo.at(ij) = v.at(ij) + eu * rand();
                }
            }
            #else
            {
                for(auto&& q: util::range(Q)) {
                    const auto efo = config::obs_error_f * feq(config::rho_ref, 0, 0, q);
                    for(auto&& ij: util::range(nxy)) {
                        fo.at(config::ijq(ij, q)) = f.at(config::ijq(ij, q)) + rand() * efo;
                    }
                }

                for(auto&& ij : util::range(nxy)) {
                    real ff[Q];
                    for(auto&& q: util::range(Q)) {
                        ff[q] = fo.at(config::ijq(ij, q));
                    }
                    const auto rr = rf(ff);
                    ro.at(ij) = rr;
                    uo.at(ij) = uf(ff, rr);
                    vo.at(ij) = vf(ff, rr);
                }
            }
            #endif  // ifdef OBSERVE_ERROR_RUV

            std::vector<real> voro(config::ny * config::nx);
            for(int j=0; j<config::ny; j++) {
                #pragma omp parallel for
                for(int i=0; i<config::nx; i++) {
                    const auto ij = config::ij(i, j);
                    if(config::is_boundary(i, j)) {
                        vor.at(ij) = 0;
                    } else {
                        const int iw = (i-1+config::nx)%config::nx;
                        const int ie = (i+1)%config::nx;
                        const int js = (j-1+config::ny)%config::ny;
                        const int jn = (j+1)%config::ny;
                        const real us = uo.at(config::ij(i, js));
                        const real un = uo.at(config::ij(i, jn));
                        const real uy = (un - us) / (2*config::dx);
                        const real vw = vo.at(config::ij(iw, j));
                        const real ve = vo.at(config::ij(ie, j));
                        const real vx = (ve - vw) / (2*config::dx);
                        voro.at(ij) = vx - uy;
                    }
                }
            }

            #ifdef OBS_XYPRUNE_NAN
            {
                for(int j=0; j<config::ny; j++) {
                    #pragma omp parallel for
                    for(int i=0; i<config::nx; i++) {
                        if(i % config::da_xyprune != 0 or j % config::da_xyprune != 0) {
                            const auto ij = config::ij(i, j);
                            ro.at(ij) = NAN;
                            uo.at(ij) = NAN;
                            vo.at(ij) = NAN;
                            voro.at(ij) = NAN;
                        }
                    }
                }
            }
            #endif

            output_file(fname0("observed",  "rho"), ro.data());
            output_file(fname0("observed",   "u"), uo.data());
            output_file(fname0("observed",   "v"), vo.data());
            output_file(fname0("observed", "vor"), voro.data());
            output_file(fname0("observed",   "f"), fo.data(), Q);
        }
    }
    #endif  // ifdef OBSERVE

    #if defined(NATURE) || defined(OBSERVE) // output nature data
    {
        if(dat.ensemble_id == 0) {
            output_file(fname0("nature",  "rho"), r.data());
            output_file(fname0("nature",   "u"), u.data());
            output_file(fname0("nature",   "v"), v.data());
            output_file(fname0("nature", "vor"), vor.data());
            output_file(fname0("nature",   "f"), f.data(), Q);
        }
    }
    #endif

    // blank linebreak
    mpi.barrier();
    if(mpi.rank() == 0 && config::verbose >= 1) {
        std::cout << std::endl;
    }

}
