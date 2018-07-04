//
//  NetworkBufferParse.cpp
//  dendrite
//
//  Created by Oli Callaghan on 16/01/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include "NetworkBufferParse.hpp"
#include "Exceptions.hpp"

// Loads input tensor size from model.struct
Dims NetworkBufferParse::LoadInput(std::ifstream* file) {
    // Load line
    std::string line;
    getline(*file, line);
    
    // Match to input tag RegExp
    std::regex input_exprn("<inp s=((?:[0-9]+)(?:,[0-9]+){3})>");
    std::smatch match;
    
    if (std::regex_match(line, match, input_exprn)) {
        // Match
        std::vector<int> dims_vec;
        
        // Extract size from string
        char* search = new char[match[1].str().length() + 1];
        std::strcpy(search, match[1].str().c_str());
        char* tok = std::strtok(search, ",");
        
        while (tok != NULL) {
            dims_vec.push_back(std::stoi(tok));
            tok = std::strtok(NULL, ",");
        }
        
        // Return Dims object
        Dims dims(dims_vec);
        return dims;
    } else {
        // Invalid line
        throw ModelStructSyntaxError(line, "<inp s=dim0,dim1,dim2,dim3>");
    }

}
