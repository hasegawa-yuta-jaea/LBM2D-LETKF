#ifndef ASSIMILATE_H_
#define ASSIMILATE_H_

#include <iostream>

#include "config.h"
#include "data.h"
#ifdef DA_LETKF
#include "util/letkf_solver.h"
#endif
#include "util/timer.hpp"
#include "util/device_memory.hpp"

class DataAssimilator {
public:
    #if defined(DOUBLE_PRECISION) || defined(LETKF_DOUBLE_PRECISION)
    using letkf_real = double;
    #else
    using letkf_real = float;
    #endif
private:
    util::device_memory<real> uo, vo, ro, fo; // observation
    std::vector<real> uo_host, vo_host, ro_host, fo_host; // observation buffer for MPI (overlapped)
    #ifdef DA_LETKF
    util::letkf_solver<letkf_real> solver;
    util::device_memory<letkf_real> xk, yk; // LETKF buffer: k-th column of X, Y
    util::device_memory<letkf_real> xk_host; // LETKF buffer for MPI (overlapped)
    #endif
    util::timer timer;
public:
    explicit DataAssimilator(const util::mpi& mpi) { init(mpi); }

private:
    void init(const util::mpi& mpi) {
        #ifdef DA_LETKF
        init_letkf(mpi);
        #elif defined(DA_NUDGING)
        init_nudging();
        #endif
    }

public: // private member function but public due to workaround of cuda_device_lambda
    void init_letkf(const util::mpi& mpi);
    void init_nudging();

public:
    void assimilate_nudging_lbm(data& dat, const int t);
    void assimilate_nudging_uv(data& dat, const int t);
    void assimilate_letkf(data& dat, const util::mpi& mpi, const int t);

public:
    // wrapper of assimilate_*()
    void assimilate(data& dat, const util::mpi& mpi, const int t) {
        #ifdef DA_LETKF
            assimilate_letkf(dat, mpi, t);
        #elif defined(DA_NUDGING)
            #if defined(DA_OBS_UV) || defined(DA_OBS_RUV)
            assimilate_nudging_uv(dat, t);
            #else
            assimilate_nudging_lbm(dat, t);
            #endif
        #endif
    }

    // getter
    const auto& get_timer() const { return timer; }
    #ifdef DA_LETKF
    const auto& get_letkf_timer() const { return solver.get_timer(); }
    #endif
    void clear_all_timer() {
        timer.clear();
        #ifdef DA_LETKF
        solver.clear_timer();
        #endif
    }

};

#endif
