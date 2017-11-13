//
//  Tensor.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <stdio.h>
#include <vector>

// Dimensions:
//  Stored as (x,y,z,batch)
struct Dims {
    int* dims;
    Dims(std::vector<int> dims);
    int Size();
    int SizePerEx();
};

class Tensor {
public:
    Dims* dims;
    float* data;
    
    Tensor(Dims d);
    Tensor(std::vector<int> d);
    void LoadData(float* data);
};

#endif /* Tensor_hpp */
