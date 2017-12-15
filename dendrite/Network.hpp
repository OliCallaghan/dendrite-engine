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
#include <OpenCL/opencl.h>
#include "Graph.hpp"
#include "Tensor.hpp"

class Network {
    Graph* g;
    Tensor* input;
    Tensor* output;
    Tensor* prediction;
    dispatch_queue_t queue;
    
public:
    float LearningRate = 0.01; // Learning rate
    
    float Learn();
    void Evaluate();
    Network(Tensor* input, Tensor* prediction, Tensor* output); // Initialise empty network
    bool Validate();
    
    bool ImportNetwork();
    bool ImportLayerParameters();
};

#endif /* Network_hpp */
