//
// Created by moinshaikh on 3/23/26.
//


#include "../include/TrainingPipeline.hpp"


namespace TrainingPipeline
{
    Pipeline::Pipeline(const std::string &path, const csv::CSVFormat &format)
    {
        csv::CSVReader reader(path, format);
        std::vector<csv::CSVRow> rows(reader.begin(),reader.end());

        bool firstRow = true;
        for (auto &row : rows)
        {
            std::vector<std::string> currentRow;
            for (auto &cell : row)
            {
                currentRow.push_back(cell.get<std::string> ());
            }
            if (firstRow)
            {
                headers = currentRow;
                firstRow = false;
            }
            else
            {
                csvData.push_back(currentRow);
            }
        }
        preValidation();
    }

    void Pipeline::preValidation()
    {
        size_t targetColCount = headers.size();
        for (auto &row  : csvData)
        {
            if (row.size() < targetColCount)
            {
                row.resize(targetColCount, "0.0");
            }
            else if (row.size() > targetColCount)
            {
                row.erase(row.begin() + targetColCount, row.end()); // Truncate extra cells
            }
        }
    }

    void Pipeline::Encoding()
    {
        int numRows= csvData.size();
        for (int col = 0; col < headers.size(); ++col)
        {
            std::string name = headers[col];
            if (name == "id") continue;

            headersTensors[name] =torch::zeros({numRows},torch::kFloat64);

            //encoding
            for (int row = 0; row < numRows; ++row)
            {
                std::string cell_value = csvData[row][col];

                if (name == "diagnosis")
                {
                    // Label encoding: M->1, B->0
                    headersTensors[name][row] = (cell_value == "M") ? 1.0f : 0.0f;
                }
                else
                {
                    // Numeric columns
                    try {
                        headersTensors[name][row] = std::stod(cell_value);
                    } catch (...) {
                        headersTensors[name][row] = 0.0f;
                    }
                }
            }
        }


        for (auto& [name, tensor] : headersTensors)
        {
            if (name != "diagnosis")
            {
                features.push_back(tensor);
            }
        }
    }

    torch::Tensor Pipeline::Normalization(torch::Tensor &X)
    {
        auto Xmean = X.mean(0 ,true);
        auto Xstd = X.std(0 ,true);
        Xstd =  torch::clamp(Xstd,1e-7);
        X= (X - Xmean) / Xstd;
        return X;
    }

    TrainTestSplit Pipeline::splitTensors(torch::Tensor X, torch::Tensor Y, float test_size)
    {
        int total_samples = X.size(0);
        int test_samples  = static_cast<int>(total_samples * test_size);
        int train_samples = total_samples - test_samples;

        auto indices       = torch::randperm(total_samples);
        auto train_indices = indices.slice(0, 0, train_samples);
        auto test_indices  = indices.slice(0, train_samples, total_samples);

        TrainTestSplit split;
        split.X_train = X.index_select(0, train_indices);
        split.X_test  = X.index_select(0, test_indices);
        split.Y_train = Y.index_select(0, train_indices).unsqueeze(1);
        split.Y_test  = Y.index_select(0, test_indices).unsqueeze(1);
        
        return split;
    }

    void Pipeline::SplitingDatasets()
    {
        // Get features and labels
        torch::Tensor X = torch::stack(features).transpose(0, 1); // Shape: [samples, features]
        torch::Tensor Y = headersTensors["diagnosis"];
        
        // Normalize features
        X = Normalization(X);
        
        // Split the dataset
        auto split = splitTensors(X, Y, 0.2);
        
        // Store results (you can add member variables to store these if needed)
        std::cout << "Training set size: " << split.X_train.size(0) << std::endl;
        std::cout << "Test set size: " << split.X_test.size(0) << std::endl;
    }




}
