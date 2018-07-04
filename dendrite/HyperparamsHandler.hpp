//
//  HyperparamsHandler.hpp
//  dendrite
//
//  Created by Oli Callaghan on 08/02/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef HyperparamsHandler_hpp
#define HyperparamsHandler_hpp

#include <string>
#include <regex>
#include <stdio.h>
#include "Layer.hpp"

// Manages the ability to save layer hyperparameters
class HyperparamsHandler {
    // Location to save to
    std::string loc;
    
public:
    HyperparamsHandler(std::string);
    void Save(std::string);
};

#endif /* HyperparamsHandler_hpp */
