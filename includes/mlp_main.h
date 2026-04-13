#include <sycl/sycl.hpp>

#ifndef MLPMain
#define MLPMain

class Layer{
    public:
        int m_n_inputs;
        int m_n_neurons;
        float* weights;
        float* biases;
        float* inputs;
        float* outputs;
        float* dweights;
        float* dbiases;
        flaot* dinputs;
        sycl::queue m_queue;
        int m_channels;
        Layer(sycl::queue q,int n_inputs, int n_neurons, int channels);

        void feedForward(float* inputs);
        void backward(float* dvalues);
};

#endif
