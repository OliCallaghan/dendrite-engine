//
//  main.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include "Tensor.hpp"
#include "Network.hpp"
#include "MNIST.hpp"

void MNISTRun(std::string str) {
    MNISTHandler MNIST("/Users/Oli/Desktop/training_data/");
    MNIST.VerifyDataSet();
    Network n(str);
    
    n.LearningRate = 1e-3;
    
    MNIST.LoadData(n.input, n.prediction);
    
    /*float loss = 0;
    for (int loop = 0; loop < 800000; loop++) {
        MNIST.LoadData(n.input, n.prediction);
        loss += n.Learn();
        if (loop % 100 == 0) {
            n.Evaluate();
            n.LearningRate = n.LearningRate * 0.9999;
            std::cout << "INPUT:\n" << n.input->GetMNISTDataStr() << "\n";
            std::cout << "ITERATION " << loop << ": " << (loss / 100) << "\n";
            loss = 0;
            std::cout << "PREDICTED: " << n.prediction->GetDataStr() << "\n";
            std::cout << "OUTPUT: " << n.output->GetDataStr() << "\n";
            if (loop % 10000 == 0) {
                n.SaveNetwork("/Users/Oli/Desktop/Network_Save/");
            }
        }
    }*/
    
    // ACCURACY[2] = {Tests, Success}
    float success = 0;
    
    for (int loop = 0; loop < 1000; loop++) {
        MNIST.LoadData(n.input, n.prediction);
        //std::cout << "INPUT:\n" << input.GetMNISTDataStr() << "\n";
        n.Evaluate();
        int prediction_class = MNIST.Classify(n.prediction);
        int output_class = MNIST.Classify(n.output);
        //std::cout << "CLASSIFY: " << prediction_class << ", " << output_class << "\n";
        if (prediction_class == output_class) {
            success += 1;
        } else {
            std::cout << "FAILED CLASSIFICATION:\n";
            std::cout << "INPUT:\n" << n.input->GetMNISTDataStr() << "\n";
            std::cout << "CLASSIFY: " << prediction_class << ", " << output_class << "\n";
        }
    }
    float accuracy_val = success / 1000;
    std::cout << "ACCURACY: " << accuracy_val << "\n";
}

void Run(std::string loc, float eta) {
    Network n(loc);
    n.LearningRate = eta;
    
    float loss = 0;
    for (int loop = 0; loop < 800000; loop++) {
        loss += n.Learn();
        if (loop % 100 == 0) {
            n.Evaluate();
            n.LearningRate = n.LearningRate * 0.9999;
            std::cout << "INPUT:\n" << n.input->GetMNISTDataStr() << "\n";
            std::cout << "ITERATION " << loop << ": " << (loss / 100) << "\n";
            loss = 0;
            std::cout << "PREDICTED: " << n.prediction->GetDataStr() << "\n";
            std::cout << "OUTPUT: " << n.output->GetDataStr() << "\n";
            if (loop % 10000 == 0) {
                n.SaveNetwork("/Users/Oli/Desktop/Network_Save/");
            }
        }
    }
}

void STDRun() {
    Tensor input({784,1,1,1});
    Tensor prediction({10,1,1,1});
    Tensor output({10,1,1,1});
    
    MNISTHandler MNIST("/Users/Oli/Desktop/training_data/");
    MNIST.VerifyDataSet();
    
    Network n(&input, &prediction, &output);
    
    n.LearningRate = 1e-2;
    
    MNIST.LoadData(&input, &prediction);
    
    float loss = 0;
    for (int loop = 0; loop < 400000; loop++) {
        MNIST.LoadData(&input, &prediction);
        loss += n.Learn();
        if (loop % 100 == 0) {
            n.Evaluate();
            n.LearningRate = n.LearningRate * 0.9999;
            std::cout << "INPUT:\n" << input.GetMNISTDataStr() << "\n";
            std::cout << "ETA: " << n.LearningRate << "\n";
            std::cout << "ITERATION " << loop << ": " << (loss / 100) << "\n";
            loss = 0;
            std::cout << "PREDICTED: " << prediction.GetDataStr() << "\n";
            std::cout << "OUTPUT: " << output.GetDataStr() << "\n";
            if (loop % 10000 == 0) {
                n.SaveNetwork("/Users/Oli/Desktop/Network_Save/");
            }
        }
    }
    
    // ACCURACY[2] = {Tests, Success}
    float success = 0;
    
    for (int loop = 0; loop < 1000; loop++) {
        MNIST.LoadData(&input, &prediction);
        //std::cout << "INPUT:\n" << input.GetMNISTDataStr() << "\n";
        n.Evaluate();
        int prediction_class = MNIST.Classify(&prediction);
        int output_class = MNIST.Classify(&output);
        //std::cout << "CLASSIFY: " << prediction_class << ", " << output_class << "\n";
        if (prediction_class == output_class) {
            success += 1;
        } else {
            std::cout << "FAILED CLASSIFICATION:\n";
            std::cout << "INPUT:\n" << input.GetMNISTDataStr() << "\n";
            std::cout << "CLASSIFY: " << prediction_class << ", " << output_class << "\n";
        }
    }
    float accuracy_val = success / 1000;
    std::cout << "ACCURACY: " << accuracy_val << "\n";
}

int main(int argc, const char * argv[]) {
    // Layers must keep standard initialisation when loaded.
    Run("/Users/Oli/Desktop/Network_Save/", 1e-5);
    //STDRun();
    return 0;
}


