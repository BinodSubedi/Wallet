#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#ifndef Math_UTIL
#define MATH_UTIL

using namespace oneapi;

class MathUtil{

    public:
        void generate_normal_matrix(sycl::queue q ,float* &store_mem, int row, int col, float mean, float stddev, int seed = 500);
        
}

#endif