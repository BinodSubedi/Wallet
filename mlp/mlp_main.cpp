#include <sycl/sycl.hpp>
#include "../includes/mlp_main.h"
#include "../includes/math_util.h"

using namespace sycl;

Layer::Layer(queue q ,int n_inputs, int n_neurons)
:m_queue{q} ,m_n_inputs{n_inputs}, m_n_neurons{n_neurons}
{

    MathUtil::generate_normal_matrix(q ,this->weights , n_neurons, n_inputs, 0, 1);
    if(this->biases == nullptr)
    {
        this->biases = malloc_device<float>(n_neurons, q);
    }
    q.fill<float>(this->biases, 0.0f, n_neurons).wait();    

}

Layer::feedForward(float* inputs)
{

    this->inputs = inputs;
    
    if(this->output == nullptr)
    {
        this->output = malloc_device<float>(this->m_n_neurons, this->q);
    }

    event cpy = q.copy<float>(this->biases, this->output,this->m_n_neurons);

    MathUtil::dot_product_add_bias(this->m_queue,math::transpose::nontrans, math::transpose::nontrans,
    this->m_n_neurons, this->m_n_inputs, this->inputs, this->weights, float* output, cpy);

}


Layer::backward(float* dvalues)
{

}