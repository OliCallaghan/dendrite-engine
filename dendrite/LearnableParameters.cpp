//
//  LearnableParameters.cpp
//  dendrite
//
//  Created by Oli Callaghan on 29/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <random>
#include <ctime>
#include "LearnableParameters.hpp"
#include "Tensor.hpp"

void LearnableParameters::InitialiseNormal(float mean, float stddev) {
    std::default_random_engine generator(static_cast<unsigned int>(time(0)));
    std::normal_distribution<float> distribution(mean, stddev);
    
    for (int loc = 0; loc < this->dims.Size(); loc++) {
        this->data[loc] = distribution(generator);
    }
}

void LearnableParameters::Update(Tensor updates) {
    // Update Weights
}
