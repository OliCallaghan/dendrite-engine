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
#include "HyperparamsHandler.hpp"

// Train :: model_location, learning_rate, learning_rate_decay_factor, iterations_to_decay, iterations_to_return_update, iterations_to_save_parameters
void Train(std::string loc, float eta, float decay, int it_decay, int it, int it_rtrn, int it_save) {
    Network n(loc);
    n.LearningRate = eta;
    
    float loss = 0;
    
    for (int loop = 0; loop < it; loop++) {
        loss += n.Learn();
        if (loop % it_rtrn == 0) {
            n.Evaluate();
            std::cout << "ITERATION " << loop << ": " << (loss / it_rtrn) << "\n";
            loss = 0;
        }
        
        if (loop % it_decay == 0) {
            n.LearningRate = n.LearningRate * decay;
        }
        
        if (loop % it_save == 0) {
            n.SaveNetwork(loc);
        }
    }
    
    n.SaveNetwork(loc);
}

void Run(std::string loc, int it, float loss_threshold) {
    Network n(loc);
    
    float success = 0;
    for (int loop = 0; loop < it; loop++) {
        n.Evaluate();
        if (n.Classify(loss_threshold)) {
            success += 1;
        }
    }
    success = success / it;
    
    std::cout << "ACCURACY: " << success * 100 << "%\n";
}

int main(int argc, const char * argv[]) {
    // dendrite run
    if (argc < 2) {
        throw "Specify more arguments";
    } else {
        if (strcmp(argv[1], "run") == 0) {
            // dendrite run
            // dendrite run /path/to/save ITERATIONS (LOSS_THRESHOLD)
            std::cout << "CALCULATING ACCURACY\n";
            if (argc == 4) {
                Run(argv[2], atoi(argv[3]), 0.1);
            } else if (argc == 5) {
                Run(argv[2], atoi(argv[3]), atof(argv[4]));
            } else {
                throw "Invalid arguments";
            }
        } else if (strcmp(argv[1], "hp_gen") == 0) {
            // Generate Hyperparameters
            // dendrite hp_gen /path/to/save PARAMS
            std::cout << "GENERATING HYPERPARAMETERS\n";
            HyperparamsHandler handler(argv[2]);
            handler.Save(argv[3]);
            // handler.Save("FC 1 nodes=9 mean=0 stddev=0.5");
        } else if (strcmp(argv[1], "train") == 0) {
            // dendrite train
            // dendrite train /path/to/save ETA ETA_DECAY IT_DECAY IT_TOTAL IT_RETRN IT_SAVE
            std::cout << "TRAINING NETWORK\n";
            if (argc != 9) {
                throw "Invalid arguments";
            }
            std::string loc = argv[2];
            float eta, eta_decay;
            int it, it_decay, it_rtrn, it_save;
            bool err = false;
            try {
                eta = atof(argv[3]);
                eta_decay = atof(argv[4]);
                it_decay = atoi(argv[5]);
                it = atoi(argv[6]);
                it_rtrn = atoi(argv[7]);
                it_save = atoi(argv[8]);
            } catch (...) {
                std::cout << "Error parsing parameters";
                err = true;
            }
            
            if (err == false) {
                Train(loc, eta, eta_decay, it_decay, it, it_rtrn, it_save);
            }
        } else {
            throw "Unknown mode";
        }
    }
    return 0;
}


