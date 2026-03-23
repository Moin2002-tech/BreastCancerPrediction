//
// Created by moinshaikh on 3/23/26.
//

#ifndef BREASTCANCERPREDICTION_TRAININGPIPELINE_HPP
#define BREASTCANCERPREDICTION_TRAININGPIPELINE_HPP

#include<iostream>
#include<torch/torch.h>
#include<csv.hpp>
#include<map>
#include<string>
#include<vector>

namespace TrainingPipeline
{
    struct TrainTestSplit
    {
        torch::Tensor X_train;
        torch::Tensor X_test;
        torch::Tensor Y_train;
        torch::Tensor Y_test;
    };

    class Pipeline
    {
    private:
    std::vector<std::string> headers= {""};
    std::vector<std::vector<std::string>> csvData = { {""} };
    std::vector<torch::Tensor> features;
    std::map<std::string, torch::Tensor> headersTensors;



    public:
        //read csv file and store it's row onto CSVROW
        explicit Pipeline(const std::string &path,const csv::CSVFormat &format);
        //extract data from csv file

        void preValidation();

        // --- STEP 3: DIRECT TENSOR CREATION AND ENCODING ---
        void Encoding();

        //Normalization
        torch::Tensor Normalization(torch::Tensor &X);

        //Train-Test Split
        TrainTestSplit splitTensors(torch::Tensor X, torch::Tensor Y, float test_size = 0.2);

        //Spliting datasets
        void SplitingDatasets();

        std::vector<std::string> getheaders() const
        {
            return headers;
        }
        std::vector<std::vector<std::string>> getcsvData() const
        {
            return csvData;
        }

        std::map<std::string, torch::Tensor> getheadersTensors() const
        {
            return headersTensors;
        }

        std::vector<torch::Tensor> getFeatures() const
        {
            return features;
        }



        // --- STEP 3: DIRECT TENSOR CREATION AND ENCODING ---






    };
}

#endif //BREASTCANCERPREDICTION_TRAININGPIPELINE_HPP