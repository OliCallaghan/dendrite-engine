//
//  LayerLoader.hpp
//  dendrite
//
//  Created by Oli Callaghan on 12/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef LayerLoader_hpp
#define LayerLoader_hpp

#include <stdio.h>
#include <iostream>
#include <regex>
#include <string>
#include <cstring>
#include <sstream>
#include "Layer.hpp"
#include "Loss.hpp"

namespace GraphLoader {
    struct LayerDetails {
        Layers::Layer_T type;
        short id;
        std::vector<short> inputs;
        std::vector<short> dependents;
    };
    bool ParseLayer(std::string line, LayerDetails*); // Outputs layer type
    Loss::Loss_T ParseLoss(std::string line);
}

#endif /* LayerLoader_hpp */
