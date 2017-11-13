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
#include <vector>

class Graph {
    std::vector<Layer> layers;
    int layer_n;
public:
    // Load graph structure **note does not load / initialise layers
    bool Load();
    bool LoadFixed(); // Until file structure is finalised
    
    // Load or initialise layers
    bool InitialiseLayers(Tensor* input);
    bool LoadLayers();
    
    void Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue);
    void Learn();
};

#endif /* Graph_hpp */
