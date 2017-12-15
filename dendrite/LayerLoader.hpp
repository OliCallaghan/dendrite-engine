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
#include <sstream>
#include "Layer.hpp"

namespace GraphLoader {
    Layer* ParseLine(std::string line); // Outputs layer type
}

#endif /* LayerLoader_hpp */
