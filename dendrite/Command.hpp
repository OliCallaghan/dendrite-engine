//
//  Command.hpp
//  dendrite
//
//  Created by Oli Callaghan on 04/03/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef Command_hpp
#define Command_hpp

#include <stdio.h>
#include <string>
#include <regex>

namespace Commands {
    // Command type
    enum Command_T {
        GetLayerDims, GetLayerData, GetLayerParams, Pause, Resume, None
    };
    
    // Command object
    struct Command {
        Command_T t;
        int p;
        Command(Command_T, int);
    };
    // Intepret command
    Command MatchCommand(std::string);
}

#endif /* Command_hpp */
