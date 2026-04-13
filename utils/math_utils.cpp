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
