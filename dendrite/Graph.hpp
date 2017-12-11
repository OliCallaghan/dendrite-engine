//
//  Graph.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Graph_hpp
#define Graph_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "LossFn.hpp"
#include <vector>

class Graph {
    std::vector<Layer> layers;
    size_t layer_n;
    Loss::LossFn* loss_fn;
public:
    // Load graph structure **note does not load / initialise layers
    bool Load();
    bool LoadFixed(); // Until file structure is finalised
    
    // Load or initialise layers
    bool InitialiseLayers(Tensor* input);
    bool LoadLayers();
    
    void Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue);
    float Learn(Tensor* input, Tensor* prediction, dispatch_queue_t* queue, float eta);
};

#endif /* Graph_hpp */
