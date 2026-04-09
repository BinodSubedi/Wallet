#include "../includes/math_util.h"

#include <oneapi/math.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace oneapi;

MathUtil::generate_normal_matrix(queue q ,float* &store_mem, int row, int col, double mean, double stddev, int seed = 500){

            size_t num_elements = row * col;

            store_mem = malloc_device<float>(num_elements, q);
            
            math::rng::philox4x32x10 engine(q,seed);

            math::rng::gaussian<float> distribution(mean, stddev);

            event rng_event = math::rng::generate(distribution,engine,num_elements, store_mem);

            rng_event.wait();


        }

MathUtil::dot_product_add_bias(
    queue q,math::transpose transpos1, math::transpose transpos2,
    int n_neurons, int n_inputs, float* input_data, float* weights, float* output, event cpy_evt)
{

    auto gemm_event = oneapi::math::blas::column_major::gemm(
    q, 
    transpos1, // Input (1 x n_inputs)
    transpos2, // Weights (n_inputs x n_neurons)
    1,          // m: rows of output
    n_neurons,  // n: cols of output
    n_inputs,   // k: inner dimension
    1.0f,       // alpha
    input_data, 1,          // Matrix A is now the Input
    weights, n_inputs, // Matrix B is now the Weights
    1.0f,       // beta
    output, 1,              // Matrix C is the Output (1 x n_neurons)
    {cpy_evt}
);

}
