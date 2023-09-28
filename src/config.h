#ifndef CONFIG_H_
#define CONFIG_H_

#define NO_DEBUG
#if defined(DEBUG) && defined(NO_DEBUG)
#error both defined: `DEBUG`, `NO_DEBUG`
#endif

#include <cmath> // M_PI

#define PORT_CUDA

#define LES // numerical viscosity by les
//#define LES_CSM
//#define ICLBM // feq with imcompressible approx.
//#define VELOCITY_FORCE_CORRECTION
//#define FORCE_SECOND_ORDER
//#define OBS_ERROR_RHO_BY_DELTA

#define EXIT_IF_NAN
#ifdef DA_LETKF
//#define NO_OUTPUT
#define DROP_FORMER_HALF_TIMER
//#define LETKF_NO_MPI_BARRIER
#endif


#if defined(LYAPNOV_NATURE) && !defined(NATURE)
#define NATURE
#endif

#if defined(DA_LETKF) || defined(DA_NUDGING)
#ifndef DATA_ASSIMILATION
#define DATA_ASSIMILATION
#endif
#endif

#ifdef DATA_ASSIMILATION
#if !defined(DA_LETKF) && !defined(DA_NUDGING)
#error select DA_LETKF or DA_NUDGING
#endif

//#define DA_OBS_UV
#define DA_OBS_RUV

#ifdef DA_LETKF
#define ENSEMBLE_STAT
#ifndef LETKF_COVINF
#define LETKF_COVINF 1
#endif
#endif

#endif // DATA_ASSIMILATION

#ifndef DAPRUNE
#define DAPRUNE 10 // default: per 200 steps at nx256; 200 ~= t_pred/10
#endif

#ifndef DA_XYPRUNE
#define DA_XYPRUNE 1
#endif

#ifndef DA_QUADRA
#define DA_QUADRA 0
#endif

#ifndef DA_NUD_RATE
#define DA_NUD_RATE 0.1
#endif

//#define DA_DUMMY // no DA; just for debug
#ifdef DA_DUMMY
#define DATA_ASSIMILATION
#endif

#define SPINUP

#define FORCE_TURB

// observation error via lbm or uv
#define OBSERVE_ERROR_RUV // in RUV mode, observation of f[q] is unavailable
#define OBS_XYPRUNE_NAN // set nan on pruned data

// log level
#define VERBOSE 20

// mesh resol
#define SCALER 4 // by which nx = 64 times
#define NX (64 * SCALER)
#ifndef IOPRUNE
#define IOPRUNE 10
#endif

// check defines
#if defined(LY_EPSILON) && !defined(LYAPNOV)
    #define LYAPNOV
#elif defined(LYAPNOV) && !defined(LY_EPSILON)
    #define LY_EPSILON (1e-8)
#endif

#if ((defined(OBSERVE) && defined(DATA_ASSIMILATION))) || \
(defined(DATA_ASSIMILATION) && defined(LYAPNOV))
    #error defining more than two of OBSERVE/DATA_ASSIMILATION/LYAPNOV are contradicted
#endif

#if ( !defined(OBSERVE) && !defined(LYAPNOV) && !defined(DATA_ASSIMILATION) && !defined(NATURE) )
    #error one of NATURE/OBSERVE/DATA_ASSIMILATION/LYAPNOV should be defined to determine run-mode
#endif

// precision of floating point vars
//#define DOUBLE_PRECISION
//#define LETKF_DOUBLE_PRECISON
#ifdef DOUBLE_PRECISION
    using real = double;
    #define MPI_REAL_T MPI_DOUBLE
#else
    using real = float;
    #define MPI_REAL_T MPI_FLOAT
#endif


#include <iostream>
#include <string>

#include "util/cuda_hostdevice.hpp"

namespace config {

constexpr auto verbose = VERBOSE; // verbose level

// god
//constexpr real Re = infty;
constexpr int ny = NX;
constexpr int nx = ny;
constexpr int scaler = SCALER;
constexpr int iiter = 5 * scaler; // output interval
#ifdef SPINUP
// XXX
constexpr int spinup = 100000 * iiter; // cold start duration
#endif
constexpr int ioprune = IOPRUNE; // sosika
constexpr int daprune = DAPRUNE; // sosika

constexpr int iter_main = iiter * 2000;
#if defined(LYAPNOV) || defined(LYAPNOV_NATURE)
constexpr int iter_total = iter_main * 5; // test longer
#elif defined(DA_DUMMY) || defined(OBSERVE)
constexpr int iter_total = iter_main + 1; // da buffer
#else
constexpr int iter_total = iter_main; // final
#endif

constexpr real cfl = 0.05;

// phys
constexpr real rho_ref = 1;
constexpr real nu = 1e-4;
constexpr real h_ref = 2*M_PI; // reference length

// infection force (Watanabe, PRE 1997)
constexpr real kf = 4; // injected wavenumber
#if SCALER == 2 // nx=128
//constexpr real fkf = 1.2;  // kf=8
//constexpr real fkf = 2.8;  // kf=4
//constexpr real fkf = 0.6;  // kf=16
//constexpr real fkf = 2.7;  // kf=4, nu=0
//constexpr real fkf = 0.55; // kf=16, nu=0
#elif SCALER == 4 // nx=256
//constexpr real fkf = 0.85; // kf=16
//constexpr real fkf = 2.1;  // kf=8
constexpr real fkf = 5.6;  // kf=4
//constexpr real fkf = 0.8;  // kf=16, nu=0
//constexpr real fkf = 5.5;  // kf=4, nu=0
#else
//constexpr real fkf = 1; // XXX : for nx=1024, kf=100
#endif


// friction (Chartkov, PRL 2007; Xia, PRE 2014)
constexpr real friction_rate = 5e-4;

constexpr real u_ref = 1;

// data assimilation parameter
constexpr int da_xyprune = DA_XYPRUNE;
constexpr int da_quadra = DA_QUADRA;
constexpr real da_nud_rate = DA_NUD_RATE;

// observation error in data assimilation
#if defined(OBSERVE) || defined(DATA_ASSIMILATION)
#   if defined(OBS_ERROR_RHO) && defined(OBS_ERROR_U)
        constexpr real obs_error_rho0 = OBS_ERROR_RHO;
        constexpr real obs_error_u = OBS_ERROR_U;
#   else
#       error OBS_ERROR_RHO and OBS_ERROR_U should be defined in makefile.macro.in
#   endif
#   ifdef OBS_ERROR_RHO_BY_DELTA
        constexpr real obs_error_rho = obs_error_rho0 * 0.0270; // 0.0270 == minmax width of rho in preliminary experiment
#   else
        constexpr real obs_error_rho = obs_error_rho0;
#   endif
#endif

// lbm d2q9
constexpr int Q = 9;
constexpr real w0 = 4./9., w1 = 1./9., w2 = 1./36.;

// cfd
#ifndef DEBUG
inline __host__ __device__ constexpr int ij(int i, int j) { return i + nx*j; }
inline __host__ __device__ constexpr int ij_periodic(int i, int j) { return ij((i+nx)%nx, (j+ny)%ny); }
inline __host__ __device__ constexpr int ijq(int i, int j, int q) { return ij(i, j) + nx*ny*q; }
inline __host__ __device__ constexpr int ijq(int ij, int q) { return ij + nx*ny*q; }
#else
inline __host__ __device__ void hd_exit_failure() { *(int*)(NULL) = -1; }
inline __host__ __device__ constexpr int ij(int i, int j) {
    if(i<0 || i>=nx || j<0 || j>=ny) {
        printf("[%s:%d:%s] index out of range: i=%d, j=%d\n", __FILE__, __LINE__, __FUNCTION__, i, j);
        hd_exit_failure();
    }
    return i + nx*j;
}
inline __host__ __device__ constexpr int ij_periodic(int i, int j) {
    return ij((i+nx)%nx, (j+ny)%ny);
}
inline __host__ __device__ constexpr int ijq(int i, int j, int q) {
    if(i<0 || i>=nx || j<0 || j>=ny || q < 0 || q >= Q) {
        printf("[%s:%d:%s] index out of range: i=%d, j=%d, q=%d\n", __FILE__, __LINE__, __FUNCTION__, i, j, q);
        hd_exit_failure();
    }
    return ij(i, j) + nx*ny*q;
}
inline __host__ __device__ constexpr int ijq(int ij, int q) {
    if(ij<0 || ij>=nx*ny || q < 0 || q >= Q) {
        printf("[%s:%d:%s] index out of range: ij=%d, q=%d\n", __FILE__, __LINE__, __FUNCTION__, ij, q);
    }
    return ij + nx*ny*q;
}
#endif

inline __host__ __device__ constexpr bool is_boundary(int i, int j) {
    //return (i == 0 || i == config::nx-1) || (j == 0 || j == config::ny-1); // cavity flow
    //return (j == 0 || j == config::ny-1); // flow past cylinder
    return false; // isotoropic turbulence
}


constexpr real c = u_ref / cfl; // lbm c_ref

constexpr real dx = h_ref / (ny-1);
constexpr real dt = dx / c;

constexpr real tau = 0.5 + 3. * nu / c/c / dt;
constexpr real omega = 1./tau;

inline __host__ __device__ constexpr int  qe(int ex, int ey) { return ex+1 + 3*(ey+1); }
inline __host__ __device__ constexpr int  ex(int q) { return (q%3)-1; }
inline __host__ __device__ constexpr int  ey(int q) { return (q/3)-1; }

inline __host__ __device__ constexpr real wq(int q) {
    int ee = ex(q) * ex(q) + ey(q) * ey(q);
    return ee == 0 ? w0 : ee == 1 ? w1 : w2;
}
inline __host__ __device__ constexpr int qr(int q) { return (Q-1-q); }

inline void inspect() {
    if(config::verbose >= 0) {
        std::cout << "config.h::" << std::endl
            << "  nx = " << nx << std::endl
            //<< "  Re = " << Re << std::endl
            << "  nu = " << nu << " m2/s" << std::endl
            << "  u_ref = " << u_ref << " m/s" << std::endl
            << "  h_ref = " << h_ref << " meter" << std::endl
            << "  Re = " << u_ref*h_ref / nu << std::endl
            << "  omega = " << omega << std::endl
            << "  dx = " << dx << " meter" << std::endl
            << "  dt = " << dt << " sec" << std::endl
            << "  iiter = " << iiter << std::endl
            << "  dt_io = " << dt*iiter << std::endl
            << "  c = " << c << " m/s" << std::endl
            << "  Area = " << dx*dx*nx*ny << std::endl
            ;
    }
}

// output
const std::string prefix = "io";

} // namespace config

#endif // CONFIG_H_
