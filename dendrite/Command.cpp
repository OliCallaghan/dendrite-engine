//
//  Command.cpp
//  dendrite
//
//  Created by Oli Callaghan on 04/03/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include "Command.hpp"
#include <iostream>

Commands::Command::Command(Commands::Command_T type, int param) {
    this->t = type;
    this->p = param;
}

Commands::Command Commands::MatchCommand(std::string cmd_str) {
    std::smatch match;
    std::regex GetLayerDimsRGX("DIMS([0-9]+)");
    std::regex GetLayerParamsRGX("PARAMS([0-9]+)");
    std::regex GetLayerDataRGX("DATA([0-9]+)");
    std::regex PauseRGX("PAUSE");
    std::regex ResumeRGX("RESUME");
    
    if (std::regex_match(cmd_str, match, GetLayerDimsRGX)) {
        return Commands::Command(Commands::Command_T::GetLayerDims, stoi(match[1]));
    } else if (std::regex_match(cmd_str, match, GetLayerParamsRGX)) {
        return Commands::Command(Commands::Command_T::GetLayerParams, stoi(match[1]));
    } else if (std::regex_match(cmd_str, match, GetLayerDataRGX)) {
        return Commands::Command(Commands::Command_T::GetLayerData, stoi(match[1]));
    } else if (std::regex_match(cmd_str, match, PauseRGX)) {
        return Commands::Command(Commands::Command_T::Pause, NULL);
    } else if (std::regex_match(cmd_str, match, ResumeRGX)) {
        return Commands::Command(Commands::Command_T::Resume, NULL);
    }
    
    return Commands::Command(Commands::Command_T::None, NULL);
}
