//
//  main.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include "Exceptions.hpp"
#include "Tensor.hpp"
#include "Network.hpp"
#include "MNIST.hpp"
#include "HyperparamsHandler.hpp"

// Train :: model_location, learning_rate, learning_rate_decay_factor, iterations_to_decay, iterations_to_return_update, iterations_to_save_parameters
void Train(std::string loc, float eta, float decay, int it_decay, int it, int it_rtrn, int it_save) {
    Network n(loc);
    n.LearningRate = eta;
    
    float loss = 0;
    
    for (int loop = 1; loop <= it; loop++) {
        loss += n.Learn();
        if (loop % it_rtrn == 0) {
            n.Evaluate();
            
            std::cout << "ITERATION " << loop << ": " << (loss / it_rtrn) << std::endl;
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

void TestPipeline(std::string location, std::string dims_str) {
    std::stringstream dims_str_strm(dims_str);
    std::string dim_str;
    std::vector<int> dims_vec;
    
    while (std::getline(dims_str_strm, dim_str, ',')) {
        dims_vec.push_back(stoi(dim_str));
    }
    
    if (dims_vec.size() != 4) {
        throw "Invalid dimension size";
    }
    
    Tensor buffer(dims_vec);
    InstructionInterpreter pipeline(location, &buffer);
    
    for (int loop = 0; loop < 5; loop++) {
        pipeline.LoadNextDataBatch();
        std::cout << buffer.GetDataStr() << std::endl;
    }
}

int main(int argc, const char * argv[]) {
    try {
        // dendrite run
        if (argc < 2) {
            throw InsufficientArguments(argc - 1, 2);
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
                    throw InsufficientArguments(argc - 1, 4);
                }
            } else if (strcmp(argv[1], "hp_gen") == 0) {
                // Generate Hyperparameters
                // dendrite hp_gen /path/to/save PARAMS
                std::cout << "GENERATING HYPERPARAMETERS\n";
                if (argc == 4) {
                    HyperparamsHandler handler(argv[2]);
                    handler.Save(argv[3]);
                } else {
                    throw InsufficientArguments(argc - 1, 3);
                }
            } else if (strcmp(argv[1], "train") == 0) {
                // dendrite train
                // dendrite train /path/to/save ETA ETA_DECAY IT_DECAY IT_TOTAL IT_RETRN IT_SAVE
                std::cout << "TRAINING NETWORK" << std::endl;
                if (argc != 9) {
                    throw InsufficientArguments(argc - 1, 8);
                }
                std::string loc = argv[2];
                float eta, eta_decay;
                int it, it_decay, it_rtrn, it_save;
                int which = 3;
                
                try {
                    eta = atof(argv[3]); which++;
                    eta_decay = atof(argv[4]); which++;
                    it_decay = atoi(argv[5]); which++;
                    it = atoi(argv[6]); which++;
                    it_rtrn = atoi(argv[7]); which++;
                    it_save = atoi(argv[8]); which++;
                } catch (...) {
                    if (which > 4) {
                        throw ConversionError(argv[which], "float");
                    } else {
                        throw ConversionError(argv[which], "integer");
                    }
                }
                
                Train(loc, eta, eta_decay, it_decay, it, it_rtrn, it_save);
            } else if (strcmp(argv[1], "test_pipeline") == 0) {
                // dendrite test_pipeline
                // dendrite test_pipeline /path/to/save INPUT|OUTPUT DIMS
                if (argc != 5) {
                    throw InsufficientArguments(argc - 1, 4);
                }
                
                std::string pipeline;
                std::stringstream location;
                location << argv[2];
                
                if (strcmp(argv[3], "input") == 0) {
                    location << "input_pipeline.instr";
                    TestPipeline(location.str(), argv[4]);
                } else if (strcmp(argv[3], "output") == 0) {
                    location << "output_pipeline.instr";
                    TestPipeline(location.str(), argv[4]);
                } else {
                    throw UnknownMode(argv[3], "input, output");
                }
            } else {
                throw UnknownMode(argv[1], "run, hp_gen, train, test_pipeline");
            }
        }
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}


