//
//  Network.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Network_hpp
#define Network_hpp

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <OpenCL/opencl.h>
#include "Graph.hpp"
#include "Tensor.hpp"
#include "NetworkBufferParse.hpp"
#include "LayerLoader.hpp"
#include "InstructionInterpreter.hpp"

// Network Class
class Network {
    Graph* g; // Graph that the network stores and train
    dispatch_queue_t queue; // OpenCL dispatch queue
    // Instruction data pipelines
    InstructionInterpreter* input_pipeline;
    InstructionInterpreter* output_pipeline;
    
public:
    // Input, output and prediction tensors
    Tensor* input;
    Tensor* output;
    Tensor* prediction;
    
    // Learning Rate
    float LearningRate = 0.01;
    
    // Training and Accuracy Testing Methods
    float Learn(); // Performs training iteration
    void Evaluate(); // Evaluates the output of a network given an input
    bool Classify(float); // Classifies the output of a network; whether it matches the exepected output
    
    // Get Layer Dims (index)
    Dims GetLayerDims(int);
    
    // Get Layer Data (index)
    Tensor* GetLayerData(int);
    
    // Get Layer Params (index)
    Tensor* GetLayerParams(int);
    
    // Constructors
    Network(Tensor* input, Tensor* prediction, Tensor* output); // Initialise empty network
    Network(std::string);
    
    // Saves the Network
    bool SaveNetwork(std::string);
};

#endif /* Network_hpp */
