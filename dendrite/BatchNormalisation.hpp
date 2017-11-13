//
//  BatchNormalisation.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef BatchNormalisation_hpp
#define BatchNormalisation_hpp

#include <stdio.h>

namespace Layers {
    namespace BatchNormalisation {
        struct Hyperparameters {
            float Momentum;
            float Epsilon;
            bool Training;
        };
    }
}

#endif /* BatchNormalisation_hpp */
