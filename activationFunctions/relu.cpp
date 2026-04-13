#include "../includes/activation.h"

#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

using namespace sycl;
using namespace oneapi;

Activation_Relu::feed_forward(queue q,float *inputs, int n_neurons)
        {
            this->inputs = inputs;
            this->m_queue = q;

            math::vm::threshold(
                this->m_queue,
                this->m_total_elements,                       // n_inputs * batch_size
                this->inputs,                                 // Input buffer
                0.0f,                                         // Threshold value
                math::vm::threshold_direction::lower, // Replace values lower than 0
                0.0f,                                         // Value to replace them with
                this->outputs                                  // Output buffer
                )
                .wait();
        }

Activation_Relu::backward(float* dvalues)
        {

            if(this->dinputs == nullptr){
                this->dinputs = malloc_device<float>(1 * this->m_n_neurons, this->m_queue);
            }

            // need to filter out elements where the inputs were 0 and pass through all the dvalues, as we just
            // propagate them to the previous layers

            this->m_queue.copy<float>(dvalues, this->dinputs, 1 * this->m_n_neurons).wait();

            float* dinputs = this->dinputs;
            float* inputs = this->inputs;

            this->m_queue.parallel_for(range{1 * this->m_n_neurons}, [=](id<1> idx)
                                       {
                                           if (inputs[idx] <= 0)
                                           {
                                               dinputs[idx] = 0;
                                           }
                                       }).wait();
        }


