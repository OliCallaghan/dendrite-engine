//
//  InstructionInterpreter.hpp
//  dendrite
//
//  Created by Oli Callaghan on 30/01/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef InstructionInterpreter_hpp
#define InstructionInterpreter_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <regex>
#include <math.h>

#include "BinaryFile.hpp"
#include "Tensor.hpp"

namespace Pipeline {
    enum Operation_T {
        DIV, MUL, ADD, READ, SOFTMAX
    };
    
    enum Data_T {
        CHAR, U_CHAR,
        INT, U_INT,
        FLOAT,
        BIT,
        NONE
    };
    
    class Operation {
    public:
        Operation_T operation;
        long bytes;
        Data_T data_type;
        float param;
        
        Operation(Operation_T, long, Data_T, float);
        
        void (*Execute)(BinaryFileHandler*, Tensor*, long, float);
    };
}

class InstructionInterpreter {
    Tensor* ten;
    std::vector<Pipeline::Operation> op_list;
    
    BinaryFileHandler* handler;
    
    std::ifstream file;
    
public:
    InstructionInterpreter(std::string loc, Tensor* buf);
    ~InstructionInterpreter();
    
    void LoadNextDataBatch();
    void OutputInstructionQueue();
    
    bool (*Classify)(Tensor*, Tensor*, float);
};

#endif /* InstructionInterpreter_hpp */
