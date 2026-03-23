//
// Created by moinshaikh on 3/24/26.
//

#ifndef BREASTCANCERPREDICTION_NNMODEL_HPP
#define BREASTCANCERPREDICTION_NNMODEL_HPP
#include<torch/torch.h>
#include<torch/nn.h>

class NNModelImpl : public torch::nn::Module
{
private:
    torch::nn::Linear linear{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
public:
    NNModelImpl(int features) : linear(torch::nn::Linear(torch::nn::LinearOptions(features, 1))), sigmoid(torch::nn::Sigmoid())
    {
        register_module("linear", linear);
        register_module("sigmoid", sigmoid);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto out = linear(x);
        out =sigmoid(out);
        return out;
    }

};
TORCH_MODULE(NNModel);

#endif //BREASTCANCERPREDICTION_NNMODEL_HPP