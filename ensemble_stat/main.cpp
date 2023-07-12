// bindiff:
// calc rmse between dir<args[0], args[1]>, for each time, {u,v}_time.dat

#include <iostream>
#include <fstream>
#include <cmath> // nan
#include <cstdio>
#include <array>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include "util/stringutils.hpp"
#include "util/argstr.hpp"
#include "util/range.hpp"
#include "util/runtime_error.hpp"

constexpr bool verbose = false;
using real = float;
using byte = char;

constexpr size_t nx = 256*256;
constexpr size_t skip = 10; // ioprune
constexpr size_t Q = 9;  // lbm

#ifndef N_ENS
#define N_ENS
#endif
constexpr int n_ens = N_ENS;

template<typename T> T sqrt_epsilon(T u) {
    return u <= 0 ? 0 : std::sqrt(u);
}

template<size_t nn>
struct buffers_t {
    using buffer_t = std::array<real, nn>;
    buffer_t buf_nature_;
    buffer_t buf_ens_mean_;
    buffer_t buf_ens_stdev_;
    std::array<buffer_t, n_ens> buf_ensemble_;
    buffers_t() {
        std::fill(buf_nature_.begin(), buf_nature_.end(), NAN);
        std::fill(buf_ens_mean_.begin(), buf_ens_mean_.end(), NAN);
        std::fill(buf_ens_stdev_.begin(), buf_ens_stdev_.end(), NAN);
        for(auto& b: buf_ensemble_) {
            std::fill(b.begin(), b.end(), NAN);
        }
    }
    real* nature() { return buf_nature_.data(); }
    real* ens_mean() { return buf_ens_mean_.data(); }
    real* ens_stdev() { return buf_ens_stdev_.data(); }
    real* ens_member(int id) { return buf_ensemble_.at(id).data(); }
};

int main(int argc, char** argv) try {
    // args: dir_left, dir_right, t
    const auto args = util::argstr(argc, argv);
    const auto prefix_nature = args.at(0);
    const auto prefix_ensemble = args.at(1);
    std::cout << "dirinfo: nature=" << prefix_nature << " -- "
        << "ensemble=" << prefix_ensemble << std::endl;

    // buf
    buffers_t<nx> buf_u, buf_v, buf_r;
    buffers_t<Q*nx> buf_f;

    // for each time
    const auto t_iter = boost::lexical_cast<int>(args.at(2));
    for(const auto t: util::range(t_iter+1)) {
        if(t % skip != 0) { continue; }
        // load nature
        {
            const std::string fu = prefix_nature + "/ens0000" + "/u_step"  + util::to_string_aligned(t, 10) + ".dat";
            const std::string fv = prefix_nature + "/ens0000" + "/v_step"  + util::to_string_aligned(t, 10) + ".dat";
            const std::string fr = prefix_nature + "/ens0000" + "/rho_step"  + util::to_string_aligned(t, 10) + ".dat";
            const std::string ff = prefix_nature + "/ens0000" + "/f_step"  + util::to_string_aligned(t, 10) + ".dat";
            std::ifstream u(fu, std::ios::binary), v(fv, std::ios::binary), r(fr, std::ios::binary), f(ff, std::ios::binary);
            runtime_assert(u.is_open(), "could not read file: " + fu);
            runtime_assert(v.is_open(), "could not read file: " + fv);
            runtime_assert(r.is_open(), "could not read file: " + fv);
            runtime_assert(f.is_open(), "could not read file: " + ff);
            u.read((byte*)buf_u.nature(), nx*sizeof(real));
            v.read((byte*)buf_v.nature(), nx*sizeof(real));
            r.read((byte*)buf_r.nature(), nx*sizeof(real));
            f.read((byte*)buf_f.nature(), Q*nx*sizeof(real));
            if(verbose) {
                std::cout << "nature@"<<t<<": "
                    << *std::max_element(buf_u.nature(), buf_u.nature()+nx) << ", "
                    << *std::max_element(buf_v.nature(), buf_v.nature()+nx) << ", "
                    << *std::max_element(buf_r.nature(), buf_r.nature()+nx) << ", "
                    << *std::max_element(buf_f.nature(), buf_f.nature()+Q*nx)
                    << std::endl;
            }
        }
        // load emsemble files
        for(const auto k: util::range(n_ens)) {
            const std::string fu = prefix_ensemble + "/ens" + util::to_string_aligned(k) + "/u_step"  + util::to_string_aligned(t, 10) + ".dat";
            const std::string fv = prefix_ensemble + "/ens" + util::to_string_aligned(k) + "/v_step"  + util::to_string_aligned(t, 10) + ".dat";
            const std::string fr = prefix_ensemble + "/ens" + util::to_string_aligned(k) + "/rho_step"  + util::to_string_aligned(t, 10) + ".dat";
            const std::string ff = prefix_ensemble + "/ens" + util::to_string_aligned(k) + "/f_step"  + util::to_string_aligned(t, 10) + ".dat";
            std::ifstream u(fu, std::ios::binary), v(fv, std::ios::binary), r(fr, std::ios::binary), f(ff, std::ios::binary);
            runtime_assert(u.is_open(), "could not read file: " + fu);
            runtime_assert(v.is_open(), "could not read file: " + fv);
            runtime_assert(r.is_open(), "could not read file: " + fr);
            runtime_assert(f.is_open(), "could not read file: " + ff);
            u.read((byte*)buf_u.ens_member(k), nx*sizeof(real));
            v.read((byte*)buf_v.ens_member(k), nx*sizeof(real));
            r.read((byte*)buf_r.ens_member(k), nx*sizeof(real));
            f.read((byte*)buf_f.ens_member(k), Q*nx*sizeof(real));
            if(verbose) {
                std::cout << "cal_ens@"<<k<<"@"<<t<<": "
                    << *std::max_element(buf_u.ens_member(k), buf_u.ens_member(k)+nx) << ", "
                    << *std::max_element(buf_v.ens_member(k), buf_v.ens_member(k)+nx) << ", "
                    << *std::max_element(buf_r.ens_member(k), buf_r.ens_member(k)+nx) << ", "
                    << *std::max_element(buf_f.ens_member(k), buf_f.ens_member(k)+Q*nx)
                    << std::endl;
            }
        }
        // calc ensemble stat
        {
            for(const auto i: util::range(nx)) {
                struct sum_t {
                    int count=0;
                    long double sum=0, ssum=0;
                    void append(long double u) {
                        count ++;
                        sum += u;
                        ssum += u*u;
                    }
                    auto mean() { runtime_assert(count == n_ens, "invalid data count"); return sum / n_ens; }
                    auto smean() { runtime_assert(count == n_ens, "invalid data count"); return ssum / n_ens; }
                    auto stdev() { return sqrt_epsilon(smean() - mean()*mean()); }
                };
                sum_t su, sv, sr, sf[Q];
                for(const auto k: util::range(n_ens)) {
                    su.append(buf_u.ens_member(k)[i]);
                    sv.append(buf_v.ens_member(k)[i]);
                    sr.append(buf_r.ens_member(k)[i]);
                    for(const auto j: util::range(Q)) {
                        sf[j].append(buf_f.ens_member(k)[j + Q*i]);
                    }
                }
                buf_u.ens_mean()[i] = su.mean();
                buf_u.ens_stdev()[i] = su.stdev();
                buf_v.ens_mean()[i] = sv.mean();
                buf_v.ens_stdev()[i] = sv.stdev();
                buf_r.ens_mean()[i] = sr.mean();
                buf_r.ens_stdev()[i] = sr.stdev();
                for(const auto j: util::range(Q)) {
                    const auto q = j + Q*i;
                    buf_f.ens_mean()[q] = sf[j].mean();
                    buf_f.ens_stdev()[q] = sf[j].stdev();
                }
            }
        }
        // calc error to nature
        {
            struct stat_t {
                int count = 0;
                long double sum = 0, spsum = 0, serr = 0, max = 0;
                void append(long double cal_mean, long double cal_stdev, long double nature) {
                    count ++;
                    sum += cal_mean;
                    spsum += cal_stdev*cal_stdev;
                    auto err = cal_mean - nature;
                    serr += err*err;
                    max = std::max(max, cal_mean);
                }
                long double spread() { return std::sqrt(spsum / count); }
                long double rmse() { return std::sqrt(serr / count); }
                long double max_debug() { return max; }
            };
            stat_t svel, srho, slbm;
            for(const auto i: util::range(nx)) {
                svel.append(buf_u.ens_mean()[i], buf_u.ens_stdev()[i], buf_u.nature()[i]);
                svel.append(buf_v.ens_mean()[i], buf_v.ens_stdev()[i], buf_v.nature()[i]);
                srho.append(buf_r.ens_mean()[i], buf_r.ens_stdev()[i], buf_r.nature()[i]);
                for(const auto j: util::range(Q)) {
                    const auto q = j + Q*i;
                    slbm.append(buf_f.ens_mean()[q], buf_f.ens_stdev()[q], buf_f.nature()[q]);
                }
            }
            runtime_assert(svel.count == nx*2, "invalid data count");
            runtime_assert(srho.count == nx, "invalid data count");
            runtime_assert(slbm.count == nx*Q, "invalid data count");
            std::cout << "t_step, spread, rmse, rho_spread, rho_rmse, f_spread, f_rmse = "
                << t
                << ", " << svel.spread()
                << ", " << svel.rmse()
                << ", " << srho.spread()
                << ", " << srho.rmse()
                << ", " << slbm.spread()
                << ", " << slbm.rmse()
                << std::endl;
            std::cerr
                << t
                << "," << svel.spread()
                << "," << svel.rmse()
                << "," << srho.spread()
                << "," << srho.rmse()
                << "," << slbm.spread()
                << "," << slbm.rmse()
                << std::endl;
        }
    }

} catch (std::runtime_error e) {
    std::cout << e.what() << std::endl;
    return 1;
} catch (std::exception e) {
    std::cout << e.what() << std::endl;
    return 2;
} catch (...) {
    std::cout << "unknown error" << std::endl;
    return 89;
} // main
