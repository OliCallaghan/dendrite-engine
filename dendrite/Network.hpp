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

class Network {
    Graph* g;
    dispatch_queue_t queue;
    InstructionInterpreter* input_pipeline;
    InstructionInterpreter* output_pipeline;
    
public:
    Tensor* input;
    Tensor* output;
    Tensor* prediction;
    
    float LearningRate = 0.01; // Learning rate
    
    float Learn();
    void Evaluate();
    Network(Tensor* input, Tensor* prediction, Tensor* output); // Initialise empty network
    Network(std::string);
    bool Validate();
    
    bool SaveNetwork(std::string);
};

#endif /* Network_hpp */
