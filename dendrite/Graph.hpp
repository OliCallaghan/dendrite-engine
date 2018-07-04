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

// Graph Class
class Graph {
    std::vector<Layer> layers; // Stores layers
    size_t layer_n; // Number of layers
    Loss::LossFn* loss_fn; // Loss function function pointer
    Loss::Loss_T loss_t; // Loss function type
public:
    bool LoadFixed(); // Until file structure is finalised
    
    bool Save(std::string, Dims, Dims); // Save network
    
    // Add layers to graph structure
    bool InsertLayer(std::string loc, Layers::Layer_T, short, std::vector<short>, std::vector<short>);
    bool InsertInput(Tensor*);
    
    // Add loss function
    void AddLoss(Loss::Loss_T);
    
    // Load or initialise layers
    bool InitialiseLayers(Tensor* input);
    bool LoadLayers(std::string);
    
    // Traiing and testing accuracy methods
    void Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue);
    float Learn(Tensor* input, Tensor* prediction, dispatch_queue_t* queue, float eta);
    
    // Get network output size
    Dims GetOutputSize();
    
    // Get Layer (index)
    Tensor* GetLayer(int, Tensor*);
    Tensor* GetLayerParams(int);
};

#endif /* Graph_hpp */
