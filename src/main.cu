#include <iostream>
#include <mpi.h>

#include <boost/lexical_cast.hpp>

#include <cuda_runtime.h>

#include "config.h"
#include "data.h"
#include "lbm.h"
#include "io.h"
#include "assimilate.h"
#include "unit.h"
#include "util/mpi_wrapper.hpp"
#include "util/argstr.hpp"
#include "util/range.hpp"
#include "util/cuda_safe_call.hpp"
#include "util/timer.hpp"

constexpr int gpu_per_node = 4;

int main(int argc, char** argv) try {
    MPI_Init(&argc, &argv);
    auto args = util::argstr(argc, argv);
    auto mpi = util::mpi(MPI_COMM_WORLD);

    auto&& timer = util::timer();

    // inspect
    if(mpi.rank() == 0 && config::verbose >= 0) {
        std::cout << "mpisize: " << mpi.size() << std::endl;
        std::cout << "args: ";
        for(auto a: args) {
            std::cout << a << " ";
        }
        std::cout << std::endl;
        config::inspect();
    }
    mpi.barrier();

    // ensemble offset
    const auto ofs_rank = 
        args.size() >= 1
        ? boost::lexical_cast<int>(args.at(0))
        : 0;

    // 
    #ifdef PORT_CUDA
    CUDA_SAFE_CALL(cudaSetDevice(mpi.rank() % gpu_per_node));
    #endif

    timer.transit("unit tests");
    unit();

    // main
    //
    timer.transit("allocate data");
    auto&& dat = data(mpi.rank() + ofs_rank);
    #ifdef DATA_ASSIMILATION
    auto&& dataAssimilator = DataAssimilator(mpi);
    #endif

    #ifdef SPINUP
    timer.transit("spinup");
    if(mpi.rank() == 0) { std::cout << "spin-up ..." << std::endl; }
    for(const auto t: util::range(config::spinup)) {
        update(dat, t - config::spinup);
        if(mpi.rank() == 0 && t % (config::spinup/100)==0) { std::cout << "." << std::flush; }
    }
    if(mpi.rank() == 0) { std::cout << std::endl << "spin-up finished" << std::endl; }
    #endif

    #ifdef LYAPNOV
    dat.d ().purturbulate(ofs_rank);
    dat.dn().purturbulate(ofs_rank);
    #endif


    timer.transit("_main");
    for(const auto tt: util::range(config::iter_total/config::iiter+1)) {

        // DA and output
        const auto t = tt * config::iiter;

        #ifdef DROP_FORMER_HALF_TIMER // to ignore first-touch overhead
        if(t == config::iter_total/2) {
            timer.clear();
            #ifdef DATA_ASSIMILATION
            dataAssimilator.clear_all_timer();
            #endif
        }
        #endif

        #if defined(DATA_ASSIMILATION) && !defined(DA_DUMMY)
        timer.transit("DA");
        if(t > 0) {
            dataAssimilator.assimilate(dat, mpi, t);
        }
        timer.transit("_sync.DA"); mpi.barrier();
        #endif

        #ifndef NO_OUTPUT
        timer.transit("output");
        output(dat, mpi, t);
        timer.transit("_sync.output"); mpi.barrier();
        #endif
         
        timer.transit("forecast");
        for(const auto t: util::range(tt * config::iiter, (tt+1) * config::iiter)) {
            const auto per = config::iiter > 100 ? config::iiter/100 : 1;
            if(mpi.rank() == 0 && config::verbose >= 1000 && t % per == 0) { std::cout << '.' << std::flush; }
            update(dat, t);
        }
        if(mpi.rank() == 0 && config::verbose >= 1000) { std::cout << std::endl; }
        timer.transit("_sync.forecast"); mpi.barrier();
    }
    timer.stop_and_ignore_latter();

    if(mpi.rank() == 0 && config::verbose >= 0) {
        std::cout << std::endl;
        #ifdef DROP_FORMER_HALF_TIMER // to ignore first-touch overhead
        std::cout << "Elapsed time for latter half loop:" << std::endl;
        #else
        std::cout << "Elapsed time:" << std::endl;
        #endif
        timer.showall(" - ");
        std::cout << std::endl;
        std::cout << "core MLUPS: " <<
            1e-6 * config::nx * config::ny * config::iter_total / timer["forecast"]
            << std::endl;
        std::cout << "total MLUPS: " <<
            1e-6 * config::nx * config::ny * config::iter_total / (timer["forecast"] + timer["output"] + timer["DA"])
            << std::endl;

        #ifdef DA_LETKF
        std::cout << "Elapsed time of DA breakdown:" << std::endl;
        dataAssimilator.get_timer().showall(" - ");
        std::cout << "  More breakdown in LETKF solver:" << std::endl;
        dataAssimilator.get_letkf_timer().showall("   * ");
        #endif
    }

    timer.fout(config::prefix + "/elapsed_time_rank" + std::to_string(mpi.rank()) + ".csv");
    #ifdef DA_LETKF
    dataAssimilator.get_timer().fout(config::prefix + "/letkf_elapsed_time_rank" + std::to_string(mpi.rank()) + ".csv");
    #endif

    MPI_Finalize();

    return 0;

} catch(std::runtime_error e) {
    std::cerr << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
}
