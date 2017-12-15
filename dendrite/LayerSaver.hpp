//
//  LayerSaver.hpp
//  dendrite
//
//  Created by Oli Callaghan on 13/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef LayerSaver_hpp
#define LayerSaver_hpp

#include <stdio.h>
#include <fstream>
#include "Layer.hpp"
#include "Loss.hpp"

namespace GraphSaver {
    bool Save(std::string loc, std::vector<Layer> layers, Loss::Loss_T loss_t);
}

#endif /* LayerSaver_hpp */
