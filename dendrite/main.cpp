//
//  main.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include "Tensor.hpp"
#include "Network.hpp"

int main(int argc, const char * argv[]) {
    Tensor input({2,1,1,1});
    Tensor prediction({1,1,1,1});
    Tensor output({1,1,1,1});
    
    // Set Input Data
    input.data[0] = 1;
    input.data[1] = 0;
    
    // Set Prediction Data
    prediction.data[0] = 1;
    
    Network n(&input, &prediction, &output);
    n.LearningRate = 0.01;
    
    float loss;
    for (int loop = 0; loop < 1000; loop++) {
        loss = n.Learn();
        std::cout << "LOSS: " << loss << "\n";
    }
    n.Evaluate();
    std::cout << "OUTPUT: " << output.GetDataStr() << "\n";
    
    return 0;
}
