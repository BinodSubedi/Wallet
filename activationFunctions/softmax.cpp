#include <sycl/sycl.hpp>
#include <oneapi/math.cpp>

#include "../includes/activation.h"

Softmax::feed_forward(queue* q, float* inputs, int n_neurons)
        {
            this->m_queue = q;
            this->m_inputs = inputs;
            this->m_n_neurons = n_neurons;

            max = malloc_device<float>(1, *m_queue);

            auto ds = mkl::stats::make_dataset<mkl::stats::layout::row_major>(1, n_neurons, inputs);

            event find_max =  mkl::stats::max(
                *m_queue,
                ds,
                this->max,
                {}
            );

            event zero_out_output, zero_out_exp_sum; 

            if(this->m_outputs == nullptr)
            {
                m_outputs = malloc_device<float>(1 * this->m_n_neurons, *(this->m_queue));
                zero_out_output = m_queue->fill<float>(this->m_outputs, 0.0f, 1 * this->m_n_neurons);
            }

            if(this->exp_sum == nullptr)
            {
                exp_sum = malloc_device<float>(1, *(this->m_queue));
                zero_out_exp_sum = m_queue->fill<float>(this->exp_sum, 0.0f, 1); 
            }

            float* in_ptr  = this->m_inputs;
            float* out_ptr = this->m_outputs;
            float* exp_ptr = this->exp_sum;
            float* max_ptr = this->max;


            event sft_e1 =  m_queue->submit([&](handler& h){
                h.depends_on(find_max);
                h.depends_on(zero_out_output);
                h.depends_on(zero_out_exp_sum);

                h.parallel_for(range{1 * m_n_neurons}, [=](id<1> idx){

                    float temp = in_ptr[idx] - max_ptr[0];
                    temp = sycl::exp(temp);

                    atomic_ref<float,
                    memory_order::relaxed,
                    memory_scope::device,
                    access::address_space::global_space> atom(exp_ptr[0]);

                    atom += temp;
                    out_ptr[idx] = temp;

                });
            });

            m_queue->submit([&](handler& h)
            {
                h.depends_on(sft_e1);

                h.parallel_for(range{1 * m_n_neurons}, [=](id<1> idx){

                    out_ptr[idx] = out_ptr[idx] / exp_ptr[0];

                });
            });

        }