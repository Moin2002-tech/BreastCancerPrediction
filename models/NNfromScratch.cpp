#include <iostream>
#include "../include/TrainingPipeline.hpp"
#include "../include/SimpleNN.hpp"
#include "../external/third_party/doctest.hpp"
// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
TEST_CASE("NNfromScratch")
{
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    TrainingPipeline::Pipeline pipeline("/home/moinshaikh/CLionProjects/BreastCancerPrediction/database/data.csv",format);
    pipeline.Encoding();

    torch::Tensor X = torch::stack(pipeline.getFeatures(), 1);
    torch::Tensor y = pipeline.getheadersTensors()["diagnosis"];

    X = pipeline.Normalization(X);

    //std::cout<<X<<std::endl;
    auto split =  pipeline.splitTensors(X, y,0.2);

    auto X_train = split.X_train;
    auto Y_train = split.Y_train;
    auto X_test = split.X_test;
    auto Y_test = split.Y_test;

    std::cout<<X_train.sizes()<<std::endl;
    std::cout<<Y_train.sizes()<<std::endl;
    std::cout<<X_test.sizes()<<std::endl;
    std::cout<<Y_test.sizes()<<std::endl;

    int epochs = 32;
    float learning_rate = 0.01;

    SimpleNN model(X_train.size(1));

    //training Loop

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Forward pass
        auto y_pred = model->forward(X_train);

        // Loss
        auto loss = model->loss_function(y_pred, Y_train);

        // Backward pass
        loss.backward();

        // Parameter update (manual, no optimizer)
        {
            torch::NoGradGuard no_grad;
            model->weight -= learning_rate * model->weight.grad();
            model->bias    -= learning_rate * model->bias.grad();
        }

        // Zero gradients
        model->weight.grad().zero_();
        model->bias.grad().zero_();

        std::cout << "Epoch: " << epoch + 1
                  << ", Loss: " << loss.item<double>() << "\n";
    }


    // ------- Evaluation on Test Set -------
    std::cout << "\n--- Test Set Evaluation ---\n";
    {
        torch::NoGradGuard no_grad;

        auto test_pred = model->forward(X_test);
        auto test_loss = model->loss_function(test_pred, Y_test);

        // Accuracy: threshold at 0.5
        auto predicted = (test_pred >= 0.5).to(torch::kFloat64);
        auto correct   = predicted.eq(Y_test).sum().item<int64_t>();
        double accuracy = static_cast<double>(correct) / X_test.size(0) * 100.0;

        std::cout << "Test Loss:     " << test_loss.item<double>() << "\n";
        std::cout << "Test Accuracy: " << accuracy << "%\n";
        std::cout << "Predictions:\n"  << test_pred << "\n";
        std::cout << "Ground truth:\n" << Y_test    << "\n";
    }
}