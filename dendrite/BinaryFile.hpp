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
#include "Exceptions.hpp"

// Binary File Handler Class
class BinaryFileHandler {
    // Loop start and endpoitns
    long end_pos;
    long start_pos;
    std::string location;
public:
    std::ifstream file; // File opened
    
    BinaryFileHandler(std::string loc);
    ~BinaryFileHandler();
    
    // Set position of binary reader
    void SetPosition(long pos);
    void Increment(long n);
    
    void Loop(long, long);
};

namespace BinaryFileReader {
    // Read N bytes from file and interpret as type
    template <class T> void ReadBytesToTensor(BinaryFileHandler* handler, Tensor* buffer, long bytes_n) {
        // Read file and loop
        handler->Increment(sizeof(T) * bytes_n);
        // Intiialise bytes buffer
        T bytes[buffer->dims.Size()];
        
        if (sizeof(T) * buffer->dims.Size() < bytes_n) {
            throw IncorrectReadSize(bytes_n / sizeof(T), buffer->dims.Size());
        }
        
        // Read from binary file
        handler->file.read(reinterpret_cast<char*>(bytes), bytes_n);
        
        // Write data to the buffer
        for (int i = 0; i < buffer->dims.Size(); i++) {
            buffer->data[i] = bytes[i];
        }
    }
    void ReadBitsToTensor(BinaryFileHandler*, Tensor*, long);
}

#endif /* BinaryFile_hpp */
