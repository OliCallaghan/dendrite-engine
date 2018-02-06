//
//  NetworkBufferParse.hpp
//  dendrite
//
//  Created by Oli Callaghan on 16/01/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef NetworkBufferParse_hpp
#define NetworkBufferParse_hpp

#include <stdio.h>
#include <fstream>
#include <regex>
#include "Tensor.hpp"

namespace NetworkBufferParse {
    Dims LoadInput(std::ifstream* file);
}

#endif /* NetworkBufferParse_hpp */
