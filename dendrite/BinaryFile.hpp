//
//  BinaryFile.hpp
//  dendrite
//
//  Created by Oli Callaghan on 29/01/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef BinaryFile_hpp
#define BinaryFile_hpp

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>

#include "Tensor.hpp"

class BinaryFileHandler {
    long end_pos;
    long start_pos;
    std::string location;
public:
    std::ifstream file;
    
    BinaryFileHandler(std::string loc);
    ~BinaryFileHandler();
    
    // Set position of binary reader
    void SetPosition(long pos);
    void Increment(long n);
    
    void Loop(long, long);
};

namespace BinaryFileReader {
    template <class T> void ReadBytesToTensor(BinaryFileHandler* handler, Tensor* buffer, long bytes_n) {
        // Read file and loop
        handler->Increment(sizeof(T) * bytes_n);
        T bytes[buffer->dims.Size()];
        handler->file.read(reinterpret_cast<char*>(bytes), bytes_n);
        
        for (int i = 0; i < buffer->dims.Size(); i++) {
            buffer->data[i] = bytes[i];
        }
    }
    void ReadBitsToTensor(BinaryFileHandler*, Tensor*, long);
}

#endif /* BinaryFile_hpp */
