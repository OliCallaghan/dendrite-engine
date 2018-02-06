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
#include "LayerLoader.hpp"
#include <vector>

class Graph {
    std::vector<Layer> layers;
    size_t layer_n;
    Loss::LossFn* loss_fn;
    Loss::Loss_T loss_t;
public:
    // Load graph structure **note does not load / initialise layers
    bool Load(std::string, Tensor*);
    bool LoadFixed(); // Until file structure is finalised
    
    bool Save(std::string, Dims, Dims);
    
    bool InsertLayer(std::string loc, Layers::Layer_T, short, std::vector<short>, std::vector<short>);
    bool InsertInput(Tensor*);
    
    void AddLoss(Loss::Loss_T);
    
    // Load or initialise layers
    bool InitialiseLayers(Tensor* input);
    bool LoadLayers(std::string);
    
    void Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue);
    float Learn(Tensor* input, Tensor* prediction, dispatch_queue_t* queue, float eta);
    
    Dims GetOutputSize();
};

#endif /* Graph_hpp */
