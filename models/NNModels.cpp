//
// Created by moinshaikh on 3/24/26.
//



#include"../include/NNModel.hpp"
#include"../include/TrainingPipeline.hpp"
#include"../external/third_party/doctest.hpp"
TEST_CASE("NNModels")
{
    std::string path = "/home/moinshaikh/CLionProjects/BreastCancerPrediction/database/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    TrainingPipeline::Pipeline pipeline(path,format);
    pipeline.Encoding();

    torch::Tensor X = torch::stack(pipeline.getFeatures(), 1);
    torch::Tensor y = pipeline.getheadersTensors()["diagnosis"];

    X = pipeline.Normalization(X);
    
    // Convert to Float32 to match model dtype
    X = X.to(torch::kFloat32);
    y = y.to(torch::kFloat32);

    auto split =  pipeline.splitTensors(X, y,0.2);

    auto X_train = split.X_train;
    auto y_train = split.Y_train;
    auto X_test = split.X_test;
    auto y_test = split.Y_test;

    int epochs = 100;
    float learning_rate = 0.01;

    NNModel model(X_train.size(1));
    auto loss = torch::nn::BCELoss();
    auto optimizer = torch::optim::SGD(model->parameters(), learning_rate);
    for (int epoch=0; epoch < epochs; ++epoch)
    {
        auto yPred = model->forward(X_train);
        auto loss1 = loss(yPred, y_train);
        optimizer.zero_grad();
        loss1.backward();
        optimizer.step();
        /*
        std::cout << "Epoch: " << epoch + 1
                 << ", Loss: " << loss1.item<double>() << "\n";*/

    }
    std::cout << "\n--- Test Set Evaluation ---\n";
    {
        torch::NoGradGuard no_grad;

        auto test_pred = model->forward(X_test);
        auto test_loss = loss(test_pred, y_test);

        // Accuracy: threshold at 0.5
        auto predicted = (test_pred >= 0.5).to(torch::kFloat32);
        auto correct   = predicted.eq(y_test).sum().item<int64_t>();
        double accuracy = static_cast<double>(correct) / X_test.size(0) * 100.0;

        std::cout << "Test Loss:     " << test_loss.item<double>() << "\n";
        std::cout << "Test Accuracy: " << accuracy << "%\n";


    }



}

