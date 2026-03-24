//
// Created for exporting model results to JSON for visualization
//

#include "include/NNModel.hpp"
#include "include/TrainingPipeline.hpp"
#include "include/NNCustumDatasets.hpp"
#include "include/SimpleNN.hpp"
#include "external/third_party/doctest.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

// Function to export model predictions to JSON
void exportModelResults(const std::string& modelName, 
                       torch::Tensor predictions,
                       torch::Tensor probabilities,
                       torch::Tensor trueLabels,
                       double accuracy,
                       double loss,
                       const std::string& filename = "model_results.json") {
    
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    
    // Start JSON object for this model
    json << "  \"" << modelName << "\": {\n";
    
    // Export predictions array - handle both Float32 and Float64
    json << "    \"predictions\": [";
    if (predictions.dtype() == torch::kFloat32) {
        auto pred_data = predictions.data_ptr<float>();
        for (int i = 0; i < predictions.numel(); ++i) {
            json << pred_data[i];
            if (i < predictions.numel() - 1) json << ", ";
        }
    } else
        {
        auto pred_data = predictions.data_ptr<double>();
        for (int i = 0; i < predictions.numel(); ++i) {
            json << pred_data[i];
            if (i < predictions.numel() - 1) json << ", ";
        }
    }
    json << "],\n";
    
    // Export probabilities array - handle both Float32 and Float64
    json << "    \"probabilities\": [";
    if (probabilities.dtype() == torch::kFloat32)
        {
        auto prob_data = probabilities.data_ptr<float>();
        for (int i = 0; i < probabilities.numel(); ++i)
            {
            json << prob_data[i];
            if (i < probabilities.numel() - 1) json << ", ";
        }
    } else {
        auto prob_data = probabilities.data_ptr<double>();
        for (int i = 0; i < probabilities.numel(); ++i)
            {
            json << prob_data[i];
            if (i < probabilities.numel() - 1) json << ", ";
        }
    }
    json << "],\n";
    
    // Export true labels array - handle both Float32 and Float64
    json << "    \"true_labels\": [";
    if (trueLabels.dtype() == torch::kFloat32) {
        auto label_data = trueLabels.data_ptr<float>();
        for (int i = 0; i < trueLabels.numel(); ++i) {
            json << label_data[i];
            if (i < trueLabels.numel() - 1) json << ", ";
        }
    } else {
        auto label_data = trueLabels.data_ptr<double>();
        for (int i = 0; i < trueLabels.numel(); ++i) {
            json << label_data[i];
            if (i < trueLabels.numel() - 1) json << ", ";
        }
    }
    json << "],\n";
    
    // Export metrics
    json << "    \"accuracy\": " << accuracy << ",\n";
    json << "    \"loss\": " << loss << "\n";
    json << "  }";
    
    // Read existing file if it exists
    std::string existing_content = "";
    std::ifstream inFile(filename);
    if (inFile.good()) {
        std::getline(inFile, existing_content, '\0');
        inFile.close();
    }
    
    // Write complete JSON
    std::ofstream outFile(filename);
    outFile << "{\n";
    
    // Add existing content (except closing brace)
    if (!existing_content.empty() && existing_content != "{}") {
        size_t last_brace = existing_content.find_last_of('}');
        if (last_brace != std::string::npos) {
            outFile << existing_content.substr(1, last_brace - 1);
            outFile << ",\n";
        }
    }
    
    // Add new model data
    outFile << json.str();
    outFile << "\n}\n";
    outFile.close();
    
    std::cout << "Exported results for " << modelName << " to " << filename << std::endl;
}

// Modified PyTorch NN Model with export functionality
void runPyTorchModelWithExport()
{
    std::string path = "/home/moinshaikh/CLionProjects/BreastCancerPrediction/database/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    TrainingPipeline::Pipeline pipeline(path, format);
    pipeline.Encoding();

    torch::Tensor X = torch::stack(pipeline.getFeatures(), 1);
    torch::Tensor y = pipeline.getheadersTensors()["diagnosis"];

    X = pipeline.Normalization(X).to(torch::kFloat32);
    y = y.to(torch::kFloat32);

    auto split = pipeline.splitTensors(X, y, 0.2);
    auto X_train = split.X_train;
    auto y_train = split.Y_train;
    auto X_test = split.X_test;
    auto y_test = split.Y_test;

    int epochs = 100;
    float learning_rate = 0.01;

    NNModel model(X_train.size(1));
    auto loss_fn = torch::nn::BCELoss();
    auto optimizer = torch::optim::SGD(model->parameters(), learning_rate);
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto yPred = model->forward(X_train);
        auto loss_val = loss_fn(yPred, y_train);
        optimizer.zero_grad();
        loss_val.backward();
        optimizer.step();
    }

    // Test evaluation and export
    torch::NoGradGuard no_grad;
    auto test_pred = model->forward(X_test);
    auto test_loss = loss_fn(test_pred, y_test);
    
    auto predicted = (test_pred >= 0.5).to(torch::kFloat32);
    auto correct = predicted.eq(y_test).sum().item<int64_t>();
    double accuracy = static_cast<double>(correct) / X_test.size(0);

    // Export results
    exportModelResults("PyTorch_NN", predicted, test_pred, y_test, 
                      accuracy, test_loss.item<double>());
}

// Modified Scratch NN Model with export functionality
void runScratchModelWithExport() 
{
    std::string path = "/home/moinshaikh/CLionProjects/BreastCancerPrediction/database/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    TrainingPipeline::Pipeline pipeline(path, format);
    pipeline.Encoding();

    torch::Tensor X = torch::stack(pipeline.getFeatures(), 1);
    torch::Tensor y = pipeline.getheadersTensors()["diagnosis"];

    X = pipeline.Normalization(X);

    auto split = pipeline.splitTensors(X, y, 0.2);
    auto X_train = split.X_train;
    auto Y_train = split.Y_train;
    auto X_test = split.X_test;
    auto Y_test = split.Y_test;

    int epochs = 32;
    float learning_rate = 0.01;

    SimpleNN model(X_train.size(1));

    // Training Loop
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
    }

    // Test evaluation
    torch::NoGradGuard no_grad;
    auto test_pred = model->forward(X_test);
    auto test_loss = model->loss_function(test_pred, Y_test);

    // Accuracy: threshold at 0.5
    auto predicted = (test_pred >= 0.5).to(torch::kFloat64);
    auto correct = predicted.eq(Y_test).sum().item<int64_t>();
    double accuracy = static_cast<double>(correct) / X_test.size(0);

    // Export results
    exportModelResults("Scratch_NN", predicted, test_pred, Y_test,
                      accuracy, test_loss.item<double>());
}

// Modified Custom Dataset Model with export functionality
void runCustomDatasetWithExport()
{
    std::string path = "/home/moinshaikh/CLionProjects/BreastCancerPrediction/database/data.csv";
    csv::CSVFormat format;
    format.delimiter(',').no_header();
    TrainingPipeline::Pipeline pipeline(path, format);
    pipeline.Encoding();

    torch::Tensor X = torch::stack(pipeline.getFeatures(), 1);
    torch::Tensor y = pipeline.getheadersTensors()["diagnosis"];

    X = pipeline.Normalization(X).to(torch::kFloat32);
    y = y.to(torch::kFloat32);

    auto split = pipeline.splitTensors(X, y, 0.2);
    auto X_train = split.X_train;
    auto y_train = split.Y_train;
    auto X_test = split.X_test;
    auto y_test = split.Y_test;

    int epochs = 100;
    float learning_rate = 0.01;

    // Create custom dataset and dataloader
    auto train_dataset = CustomDatasets(X_train, y_train)
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(32));

    NNModel model(X_train.size(1));
    auto loss_fn = torch::nn::BCELoss();
    auto optimizer = torch::optim::SGD(model->parameters(), learning_rate);

    // Training loop with batches
    for (int epoch = 0; epoch < epochs; ++epoch)
        {
        for (auto& batch : *train_loader)
            {
            auto data = batch.data;
            auto targets = batch.target.view({-1, 1});
            
            auto yPred = model->forward(data);
            auto loss_val = loss_fn(yPred, targets);
            optimizer.zero_grad();
            loss_val.backward();
            optimizer.step();
        }
    }

    // Test evaluation
    torch::NoGradGuard no_grad;
    auto test_pred = model->forward(X_test);
    auto test_loss = loss_fn(test_pred, y_test);
    
    auto predicted = (test_pred >= 0.5).to(torch::kFloat32);
    auto correct = predicted.eq(y_test).sum().item<int64_t>();
    double accuracy = static_cast<double>(correct) / X_test.size(0);

    // Export results
    exportModelResults("Custom_Dataset", predicted, test_pred, y_test,
                      accuracy, test_loss.item<double>());
}

TEST_CASE("exportData")
{

    std::remove("model_results.json");
    runPyTorchModelWithExport();
    runScratchModelWithExport();
    runCustomDatasetWithExport();
    
    std::cout << "All model results exported to model_results.json\n";
    std::cout << "Run 'python visualize_models.py' to generate visualizations\n";
}
