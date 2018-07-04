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
#include <string>
#include "Exceptions.hpp"

// Dimensions:
//  Stored as (x,y,z,batch)
struct Dims {
    int* dims;
    Dims(std::vector<int> dims);
    int Size() const;
    int SizePerEx();
    std::string GetSizeStr();
    std::string GetSizeStr(std::string);
};

// Tensor primitive
class Tensor {
public:
    Dims dims;
    float* data;
    
    Tensor(Dims d);
    Tensor(std::vector<int> d);
    std::string GetDataStr() const;
    std::string GetMNISTDataStr();
};

#endif /* Tensor_hpp */
