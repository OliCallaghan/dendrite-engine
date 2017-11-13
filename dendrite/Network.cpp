//
//  Network.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include "Network.hpp"

Network::Network(Tensor* i, Tensor* p) {
    this->g = new Graph();
    this->g->LoadFixed();
    
    this->input = i;
    this->prediction = p;
    
    this->g->InitialiseLayers(this->input);
    
    // Initialise OpenCL dispatch queue
    this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    if (this->queue == NULL) {
        // Revert to CPU if GPU is unavailable
        this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
}

void Network::Evaluate() {
    this->g->Evaluate(this->input, this->prediction, &(this->queue));
}
