#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>


using namespace sycl;
using namespace oneapi;

#ifndef RELU_ACTIVATION
#define RELU_ACTIVATION

class Activation_Relu{
    
    public:
        float* inputs;
        float* outputs;
        float* dinputs;
        queue* m_queue;
        int m_n_neurons;

        void feed_forward(queue q,float *inputs, int n_neurons);
        void backward(float* dvalues);
};

#endif