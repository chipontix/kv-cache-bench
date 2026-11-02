// Length boundaries of the paged cache: empty, 1, exactly a page, a partition or the capacity, and
// one past them. The grid of gpu_tests has no length that is a multiple of a page or a partition,
// and no empty sequence. Exit code 77 (CTest "skipped") when there is no Metal device; the checks
// that need no GPU have run by then.
#include <cmath>
#include <cstdio>

#include "workload.hpp"

using namespace kvb;

namespace {

int checks = 0, failed = 0;

void expect(bool ok, const char* what) {
    ++checks;
    if (ok) return;
    ++failed;
    std::printf("FAIL %s\n", what);
}

int finish() {
    std::printf("%d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}

}  // namespace

int main() {
    {  // The comparison every test goes through: a NaN is a difference, wherever it is.
        const float a[] = {1.0f, NAN, 3.0f}, b[] = {1.0f, 2.0f, 3.0f}, c[] = {1.0f, 2.0f, 3.5f};
        expect(std::isinf(max_abs_diff(a, b, 3)), "max_abs_diff: NaN on one side counts as infinite");
        expect(std::isinf(max_abs_diff(a, a, 3)), "max_abs_diff: NaN on both sides counts as infinite");
        expect(max_abs_diff(b, c, 3) == 0.5, "max_abs_diff: finite values");
    }
    return finish();
}
