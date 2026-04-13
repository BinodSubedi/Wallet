#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cmath>


using namespace sycl;
using namespace oneapi;

#ifndef ACTIVATION_FUNC
#define ACTIVATION_FUNC

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


class Softmax{
    queue* m_queue;
    float* m_inputs;
    int m_n_neurons;
    float* m_outputs = nullptr;
    float* exp_sum = nullptr;
    float* max = nullptr;
    public:
        feed_forward(queue* q, float* inputs, int n_neurons);

        ~Softmax(){
            sycl::free(m_outputs, *m_queue);
            sycl::free(exp_sum, *m_queue);
            sycl::free(max, *m_queue);
        }
}

#endif