//
// Created by moinshaikh on 3/24/26.
//

#include"../include/NNCustumDatasets.hpp"
#include"../external/third_party/doctest.hpp"
#include"../include/TrainingPipeline.hpp"
TEST_CASE("NNCustumDatasets")
{
    std::string path = "/home/moinshaikh/CLionProjects/BreastCancerPrediction/database/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    TrainingPipeline::Pipeline pipeline(path, format);
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

    auto train_dataset = CustomDatasets(X_train,y_train).map(torch::data::transforms::Stack<>());
    auto test_dataset = CustomDatasets(X_test,y_test).map(torch::data::transforms::Stack<>());
    std::cout<<"Training set size: "<<*train_dataset.size()<<"\n";
    std::cout<<"Test set size: "<<*test_dataset.size()<<"\n";

    int batchsize = 32;

    // 2. Use std::move() to transfer ownership to the DataLoader
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batchsize)
    );

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(batchsize)
    );

float learningRate = 0.01;
    int epochs = 25;

    auto model = NNModelCustom(X_train.size(1));
    auto optimizer = torch::optim::SGD(model->parameters(), learningRate);
    auto lossFunction = torch::nn::BCELoss();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model->train();
        for (auto batch : *train_loader) {
            auto batchFeature = batch.data;
            auto batchLabel   = batch.target;

            optimizer.zero_grad();
            auto yPred = model->forward(batchFeature);
            auto loss  = lossFunction(yPred, batchLabel);
            loss.backward();
            optimizer.step();

            std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<double>() << "\n";
        }
    }
    std::cout << "\n--- Test Set Evaluation ---\n";
    {
        torch::NoGradGuard no_grad;
        model->eval();

        double total_loss = 0.0;
        int64_t correct_preds = 0;
        int64_t total_samples = 0;
        int64_t num_batches = 0; // Backup counter for the average

        for (auto& batch : *test_loader) {
            auto data    = batch.data;
            auto targets = batch.target;

            auto output = model->forward(data);  // already has sigmoid inside


            auto loss_tensor = torch::binary_cross_entropy(output, targets);
            total_loss += loss_tensor.item<double>();

            auto predicted = (output >= 0.5).to(torch::kInt64);
            correct_preds += predicted.reshape_as(targets).eq(targets).sum().item<int64_t>();
            total_samples += data.size(0);
            num_batches++;
        }

        // FIX: value_or() needs an argument, e.g., 1 to avoid division by zero
        // Or better yet, use the manual num_batches we just calculated
        double avg_loss = total_loss / (num_batches > 0 ? num_batches : 1);
        double accuracy = (total_samples > 0) ? (static_cast<double>(correct_preds) / total_samples * 100.0) : 0.0;

        std::cout << "Test Loss:     " << avg_loss << "\n";
        std::cout << "Test Accuracy: " << accuracy << "%\n";
    }
}
