//
// Created by moinshaikh on 3/24/26.
//



#ifndef BREASTCANCERPREDICTION_SIMPLENN_HPP
#define BREASTCANCERPREDICTION_SIMPLENN_HPP

#include<torch/torch.h>
#include<torch/nn.h>

class SimpleNNImpl : public torch::nn::Module
{
public:

    torch::Tensor weight;
    torch::Tensor bias;
    SimpleNNImpl(int features)
    {
        weight =register_parameter("weight",torch::rand({features,1},torch::dtype(torch::kFloat64)));
        bias =  register_parameter("bias",torch::zeros(1,torch::dtype(torch::kFloat64)));
    }

    torch::Tensor forward(torch::Tensor X) {
        auto z = torch::matmul(X, weight) + bias;
        return torch::sigmoid(z);
    }

    torch::Tensor loss_function(torch::Tensor y_pred, torch::Tensor y) {
        const double epsilon = 1e-7;
        y_pred = torch::clamp(y_pred, epsilon, 1.0 - epsilon);
        auto loss = -(y * torch::log(y_pred) +
                     (1 - y) * torch::log(1 - y_pred)).mean();
        return loss;
    }

};
TORCH_MODULE(SimpleNN);

#endif //BREASTCANCERPREDICTION_SIMPLENN_HPP