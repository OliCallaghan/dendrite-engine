//
//  main.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include <thread>
#include <atomic>
#include "Exceptions.hpp"
#include "Tensor.hpp"
#include "Network.hpp"
#include "MNIST.hpp"
#include "HyperparamsHandler.hpp"
#include "Command.hpp"

void GetCommand(std::atomic<Commands::Command> &cmd, std::atomic<bool> &get) {
    std::string cmd_str;
    while (get.load()) {
        std::cin >> cmd_str;
        
        try {
            Commands::Command cmd_tmp = Commands::MatchCommand(cmd_str);
            cmd.store(cmd_tmp);
        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }
}

// Train :: model_location, learning_rate, learning_rate_decay_factor, iterations_to_decay, iterations_to_return_update, iterations_to_save_parameters
void Train(std::string loc, float eta, float decay, int it_decay, int it, int it_rtrn, int it_save) {
    Network n(loc);
    n.LearningRate = eta;
    
    float loss = 0;
    std::atomic<bool> get(true);
    std::atomic<Commands::Command> cmd(Commands::Command(Commands::Command_T::None, NULL));
    std::thread cmdInThread(GetCommand, std::ref(cmd), std::ref(get));
    
    int loop = 1;
    bool train = true;
    try {
        while (loop <= it) {
            if (train) {
                loss += n.Learn();
                if (loop % it_rtrn == 0) {
                    if (isnan(loss) || isinf(loss)) {
                        throw NaNException();
                    }
                    n.Evaluate();
                    std::cout << n.output->GetDataStr();
                    std::cout << n.prediction->GetDataStr();
                    std::cout << "ITERATION " << loop << ": " << (loss / it_rtrn) << std::endl;
                    loss = 0;
                }
                
                if (loop % it_decay == 0) {
                    n.LearningRate = n.LearningRate * decay;
                }
                
                if (loop % it_save == 0) {
                    n.SaveNetwork(loc);
                }
                
                loop++;
            }
            
            if ((cmd.load()).t != Commands::Command_T::None) {
                Commands::Command cmd_cpy = cmd.load();
                try {
                    switch (cmd_cpy.t) {
                        case Commands::Command_T::GetLayerDims:
                            // Get layer dims
                            std::cout << "DIMS: [ " << n.GetLayerDims(cmd_cpy.p).GetSizeStr() << " ]" << std::endl;
                            break;
                        case Commands::Command_T::GetLayerData:
                            std::cout << "DATA: " << n.GetLayerData(cmd_cpy.p)->GetDataStr() << std::endl;
                            break;
                        case Commands::Command_T::GetLayerParams:
                            std::cout << "PARAMS: " << n.GetLayerParams(cmd_cpy.p)->GetDataStr() << std::endl;
                            break;
                        case Commands::Command_T::Pause:
                            train = false;
                            break;
                        case Commands::Command_T::Resume:
                            train = true;
                            break;
                        default:
                            throw UnknownMode("UNKNOWN", "DIMS, DATA, PAUSE, RESUME");
                            break;
                    }
                } catch (std::exception &e) {
                    std::cerr << e.what();
                }
                cmd.store(Commands::Command(Commands::Command_T::None, NULL));
            }
        }
        
        get.store(false);
        
        n.SaveNetwork(loc);
        exit(0);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        exit(SIGTERM);
    }
}

void Run(std::string loc, int it, float loss_threshold) {
    Network n(loc);
    
    float success = 0;
    for (int loop = 0; loop < it; loop++) {
        if (loop % 25 == 0) {
            std::cout << ((100 * loop) / it) << "% COMPLETE" << std::endl;
        }
        n.Evaluate();
        if (n.Classify(loss_threshold)) {
            success += 1;
        }
    }
    success = success / it;
    
    std::cout << "ACCURACY: " << success * 100 << "%" << std::endl;
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
                std::cout << "CALCULATING ACCURACY" << std::endl;
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
                std::cout << "GENERATING HYPERPARAMETERS" << std::endl;
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


