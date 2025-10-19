#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
using us = std::chrono::microseconds;

struct RunStats {
    std::string label;
    std::size_t n{};
    long long time_us{};
    long long result{}; // max |a[i]-a[i-1]| (i>=1)
};

// Median of N repeats (N small)
class Timer {
public:
    void start() { t0 = Clock::now(); }
    long long stop_us() const {
        auto t1 = Clock::now();
        return std::chrono::duration_cast<us>(t1 - t0).count();
    }
private:
    Clock::time_point t0{};
};

static inline long long llabsll(long long x) {
#if defined(_MSC_VER)
    return static_cast<long long>(std::llabs(x));
#else
    return std::llabs(x);
#endif
}

// 1) Library algorithm without execution policy: std::adjacent_difference
long long max_adjacent_diff_library_no_policy(const std::vector<long long>& a) {
    if (a.size() <= 1) return 0LL;
    std::vector<long long> diff(a.size());
    std::adjacent_difference(a.begin(), a.end(), diff.begin());
    // diff[0] = a[0]; diffs from index 1 are a[i]-a[i-1]
    long long mx = 0;
    for (std::size_t i = 1; i < diff.size(); ++i) {
        long long v = llabsll(diff[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

// 2) Equivalent operation implemented with std::transform (+ policies)
//    This allows testing seq / par / par_unseq policies.

template <class ExecPolicy>
long long max_adjacent_diff_with_policy(const std::vector<long long>& a, ExecPolicy&& policy) {
    if (a.size() <= 1) return 0LL;
    const std::size_t n = a.size();
    std::vector<long long> diff(n - 1);
    // Compute diff[i] = |a[i+1] - a[i]| for i in [0..n-2]
    std::transform(policy, a.begin() + 1, a.end(), a.begin(), diff.begin(),
        [](long long curr, long long prev) {
            return llabsll(curr - prev);
        });
    // Reduce to max
    return std::reduce(policy, diff.begin(), diff.end(), 0LL, [](long long x, long long y) {
        return x > y ? x : y;
        });
}

// 3) Custom parallel algorithm: split the pair-index domain [1..n-1] into K chunks
long long max_adjacent_diff_custom_parallel(const std::vector<long long>& a, unsigned K) {
    if (a.size() <= 1) return 0LL;
    const std::size_t n = a.size();
    if (K == 0) K = 1;
    K = std::min<unsigned>(K, static_cast<unsigned>(n - 1));

    std::vector<long long> local_max(K, 0);
    std::vector<std::thread> threads;
    threads.reserve(K);

    auto worker = [&](unsigned idx, std::size_t start_pair, std::size_t end_pair) {
        // start_pair..end_pair inclusive over i in [1..n-1] (pairs are (i-1,i))
        long long mx = 0;
        for (std::size_t i = start_pair; i <= end_pair; ++i) {
            long long v = llabsll(a[i] - a[i - 1]);
            if (v > mx) mx = v;
        }
        local_max[idx] = mx;
        };

    // Partition the (n-1) pair indices roughly evenly across K chunks
    const std::size_t total_pairs = n - 1;
    const std::size_t base = total_pairs / K;
    std::size_t rem = total_pairs % K; // first 'rem' chunks get +1

    std::size_t next = 1; // first pair index
    for (unsigned t = 0; t < K; ++t) {
        std::size_t len = base + (rem ? 1 : 0);
        if (rem) --rem;
        std::size_t l = next;
        std::size_t r = l + (len ? len - 1 : 0);
        if (len == 0) { local_max[t] = 0; continue; }
        next = r + 1;
        threads.emplace_back(worker, t, l, r);
    }
    for (auto& th : threads) if (th.joinable()) th.join();

    // Sequential reduction of K local maxima
    long long mx = 0;
    for (long long v : local_max) if (v > mx) mx = v;
    return mx;
}

// Random data generator
std::vector<long long> make_data(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    // Wide uniform range to make differences meaningful.
    std::uniform_int_distribution<long long> dist(-9'000'000'000LL, 9'000'000'000LL);
    std::vector<long long> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Measure helper: run fun() R repeats, return median time (us) and result value from the last run
template <class F>
std::pair<long long, long long> measure_us(const F& fun, int repeats = 3) {
    std::vector<long long> times;
    times.reserve(repeats);
    Timer tim;
    long long lastVal = 0;
    for (int i = 0; i < repeats; ++i) {
        tim.start();
        lastVal = fun();
        times.push_back(tim.stop_us());
    }
    std::sort(times.begin(), times.end());
    return { times[times.size() / 2], lastVal };
}

int main(int argc, char** argv) {
    // Configuration (can be overridden via CLI): sizes, repeats, maxK multiplier
    std::vector<std::size_t> sizes = { 100'000, 1'000'000, 5'000'000 };
    int repeats = 3;
    double k_multiplier = 2.0; // test K up to 2x hardware threads

    if (argc >= 2) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(static_cast<std::size_t>(std::stoull(argv[i])));
        }
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    std::cout << "Build info:\n";
#if defined(NDEBUG)
    std::cout << "  NDEBUG defined (Release-like build)\n";
#else
    std::cout << "  NDEBUG NOT defined (Debug-like build)\n";
#endif
#if defined(__GNUC__)
    std::cout << "  Compiler: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n";
#elif defined(__clang__)
    std::cout << "  Compiler: Clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "  Compiler: MSVC " << _MSC_VER << "\n";
#else
    std::cout << "  Compiler: (unknown)\n";
#endif
    std::cout << "  Hardware threads: " << hw << "\n\n";

    for (std::size_t n : sizes) {
        std::cout << "=== n = " << n << " ===\n";
        auto data = make_data(n, /*seed*/ 123456789ULL + n);

        // 1) std::adjacent_difference (no policy)
        auto [t1, r1] = measure_us([&] { return max_adjacent_diff_library_no_policy(data); }, repeats);
        std::cout << "std::adjacent_difference (no policy): " << t1 << " us, max= " << r1 << "\n";

        // 2) With policies via transform/reduce
        auto [t2s, r2s] = measure_us([&] {
            return max_adjacent_diff_with_policy(data, std::execution::seq);
            }, repeats);
        std::cout << "transform+reduce [seq]: " << t2s << " us, max= " << r2s << "\n";

        auto [t2p, r2p] = measure_us([&] {
            return max_adjacent_diff_with_policy(data, std::execution::par);
            }, repeats);
        std::cout << "transform+reduce [par]: " << t2p << " us, max= " << r2p << "\n";

        auto [t2u, r2u] = measure_us([&] {
            return max_adjacent_diff_with_policy(data, std::execution::par_unseq);
            }, repeats);
        std::cout << "transform+reduce [par_unseq]: " << t2u << " us, max= " << r2u << "\n";

        // 3) Custom parallel algorithm across K chunks
        const unsigned Kmax = std::max(1u, static_cast<unsigned>(hw * k_multiplier));
        std::cout << "\nK, time_us, max_abs_diff (custom parallel)\n"; // CSV-like table header
        long long best_time = std::numeric_limits<long long>::max();
        unsigned best_K = 1;
        long long best_val = 0;
        for (unsigned K = 1; K <= Kmax; ++K) {
            auto [tk, rk] = measure_us([&] { return max_adjacent_diff_custom_parallel(data, K); }, repeats);
            std::cout << K << ", " << tk << ", " << rk << "\n";
            if (tk < best_time) { best_time = tk; best_K = K; best_val = rk; }
        }
        std::cout << "\nBest K: " << best_K << "; best time: " << best_time << " us; max_abs_diff: " << best_val << "\n";
        std::cout << "Relation to hardware threads (K/hw): " << std::fixed << std::setprecision(2)
            << (static_cast<double>(best_K) / hw) << " (hw=" << hw << ")\n\n";
    }

    std::cout << "Notes:\n";
    std::cout << " - Compile in RELEASE for measurements. Compare /Od vs /O2 (MSVC) or -O0 vs -O3 (GCC/Clang).\n";
    std::cout << " - For policy tests: std::adjacent_difference has no policy overload, so an equivalent operation\n";
    std::cout << "   is benchmarked via transform+reduce with execution policies (seq/par/par_unseq).\n";
    std::cout << " - The custom algorithm partitions the pair domain [1..n-1] into K chunks and reduces maxima.\n";
    return 0;
}
