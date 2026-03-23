//
// Created by moinshaikh on 3/24/26.
//




#ifndef BREASTCANCERPREDICTION_NNCUSTUMDATASETS_HPP
#define BREASTCANCERPREDICTION_NNCUSTUMDATASETS_HPP

#include<torch/torch.h>
#include<torch/nn.h>

class CustomDatasets: public torch::data::datasets::Dataset<CustomDatasets>
{
private:
    torch::Tensor features;
    torch::Tensor labels;
public:
    CustomDatasets(torch::Tensor features, torch::Tensor labels) : features(features), labels(labels) {}
    // Now this will correctly link with your Dataset definition
    torch::data::Example<> get(size_t index) override
    {
        return {features[index], labels[index]};
    }

    torch::optional<size_t> size() const override
    {
        return features.size(0);
    }
};


class NNModelCustomImpl : public torch::nn::Module {
public:
    torch::nn::Linear linear{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};

    NNModelCustomImpl(int features) : linear(torch::nn::Linear(features, 1)), sigmoid(torch::nn::Sigmoid()) {
        register_module("linear", linear);
        register_module("sigmoid", sigmoid);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto out = linear(x);
        out = sigmoid(out);
        return out;
    }
};
TORCH_MODULE(NNModelCustom);
#endif //BREASTCANCERPREDICTION_NNCUSTUMDATASETS_HPP