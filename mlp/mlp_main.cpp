#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include "../includes/mlp_main.h"
#include "../includes/math_util.h"

using namespace sycl;

Layer::Layer(queue q ,int n_inputs, int n_neurons, int channels)
:m_queue{q} ,m_n_inputs{n_inputs}, m_n_neurons{n_neurons}, m_channels{channels} 
{

    MathUtil::generate_normal_matrix(q ,this->weights, n_inputs, n_neurons, 0, 1);
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

    auto gemm_event = oneapi::math::blas::column_major::gemm(
    this->m_queue, 
    math::transpose::nontrans, // Input (1 x n_inputs)
    math::transpose::nontrans, // Weights (n_inputs x n_neurons)
    1,          // m: rows of output
    this->m_n_neurons,  // n: cols of output
    this->m_n_inputs,   // k: inner dimension
    1.0f,       // alpha
    this->inputs, 1,          // Matrix A is now the Input
    this->weights, this->m_n_inputs, // Matrix B is now the Weights
    1.0f,       // beta
    output, 1,              // Matrix C is the Output (1 x n_neurons)
    {cpy}
    ).wait();

}


void Layer::backward(float* dvalues) {
    /* * Operation: dweights = inputs.T @ dvalues
     * * MATH SHAPES (Logical):
     * ----------------------
     * Matrix A (inputs):     (1 x n_inputs)  -> Transposed to (n_inputs x 1)
     * Matrix B (dvalues):    (1 x n_neurons)
     * Matrix C (dweights):   (n_inputs x n_neurons)
     * * ONEAPI GEMM DIMENSIONS:
     * -----------------------
     * m = n_inputs    (Rows of Op(A) and Rows of C)
     * n = n_neurons   (Cols of Op(B) and Cols of C)
     * k = 1           (Cols of Op(A) and Rows of Op(B) - the "shared" dimension)
     */

    oneapi::math::blas::column_major::gemm(
        this->m_queue, 
        oneapi::math::transpose::trans,    // Op(A): Transpose (1 x n_inputs) to (n_inputs x 1)
        oneapi::math::transpose::nontrans, // Op(B): Keep (1 x n_neurons) as is
        this->m_n_inputs,                  // m: Rows of result (dweights)
        this->m_n_neurons,                 // n: Cols of result (dweights)
        1,                                 // k: Inner dimension (Batch Size)
        1.0f,                              // alpha
        this->inputs, 1,                   // Matrix A: (1 row, n_inputs cols), LDA = 1
        dvalues, 1,                        // Matrix B: (1 row, n_neurons cols), LDB = 1
        0.0f,                              // beta
        this->dweights, this->m_n_inputs,  // Matrix C: (n_inputs rows, n_neurons cols), LDC = n_inputs
        {}
    ).wait();


    // similarly need to do this for dbiases and dinputs

    // for dbaises, it will simply be dvalues, as the dy/db will be 1 and using chain rule the change is dvalues
    if(this->dbiases == nullptr){
        this->dbiases = malloc_device<float>(1 * this->m_n_neurons, this->m_queue);
    }

    this->m_queue.copy<float>(dvalues, this->dbiases, 1 * this->m_n_neurons).wait();

    // // For dinputs now

    //  oneapi::math::blas::column_major::gemm(
    //     this->m_queue, 
    //     oneapi::math::transpose::nontrans,    // Op(A): Transpose (1 x n_inputs) to (n_inputs x 1)
    //     oneapi::math::transpose::trans, // Op(B): Keep (1 x n_neurons) as is
    //     1,                  // m: Rows of result (dweights)
    //     this->m_n_neurons,                 // n: Cols of result (dweights)
    //     1,                                 // k: Inner dimension (Batch Size)
    //     1.0f,                              // alpha
    //     dvalues, 1,                   // Matrix A: (1 row, n_inputs cols), LDA = 1
    //     this->weights, this->m_n_inputs,                        // Matrix B: (1 row, n_neurons cols), LDB = 1
    //     0.0f,                              // beta
    //     this->dinputs, 1,  // Matrix C: (n_inputs rows, n_neurons cols), LDC = n_inputs
    //     {}
    // ).wait();   

    oneapi::math::blas::column_major::gemm(
    this->m_queue, 
    oneapi::math::transpose::nontrans, // Matrix A (dvalues) is (1 x n_neurons)
    oneapi::math::transpose::trans,    // Matrix B (weights) transpose (n_inputs x n_neurons) -> (n_neurons x n_inputs)
    1,                                 // m: Rows of result (dinputs is 1 row)
    this->m_n_inputs,                  // n: Cols of result (dinputs has n_inputs cols)
    this->m_n_neurons,                 // k: Inner dimension (n_neurons)
    1.0f,                              // alpha
    dvalues, 1,                        // Matrix A (dvalues): 1 row, LDA = 1
    this->weights, this->m_n_inputs,   // Matrix B (weights): m_n_inputs rows, LDB = m_n_inputs
    0.0f,                              // beta
    this->dinputs, 1,                  // Matrix C (dinputs): 1 row, LDC = 1
    {}
    ).wait();

}