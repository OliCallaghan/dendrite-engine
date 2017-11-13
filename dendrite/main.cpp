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
    Tensor input({9,1,1,1});
    input.data[0] = 1;
    input.data[1] = 2;
    input.data[2] = 1;
    input.data[3] = 2;
    input.data[4] = 1;
    input.data[5] = 2;
    input.data[6] = 1;
    input.data[7] = 2;
    input.data[8] = 1;
    
    Tensor output({9,1,1,1});
    
    Network n(&input, &output);
    n.Evaluate();
    std::cout << output.data[0];
    std::cout << "Hello, World!\n";
    return 0;
}
