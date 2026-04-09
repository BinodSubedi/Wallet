#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#ifndef Math_UTIL
#define MATH_UTIL

using namespace oneapi;

class MathUtil{

    public:
        void generate_normal_matrix(sycl::queue q ,float* &store_mem, int row, int col, float mean, float stddev, int seed = 500);
        void dot_product_add_bias(queue q,math::transpose transpos1, math::transpose transpos2,
                                  int n_neurons, int n_inputs, float* input_data, 
                                  float* weights, float* output, event cpy_evt);
}

#endif