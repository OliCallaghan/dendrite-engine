//
//  Network.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "Network.hpp"

Network::Network(Tensor* i, Tensor* p, Tensor* o) {
    this->g = new Graph();
    this->g->LoadFixed();
    
    // learning F(i) -> o; p is output of network
    this->input = i;
    this->prediction = p;
    this->output = o;
    
    this->g->InitialiseLayers(this->input);
    
    // Initialise OpenCL dispatch queue
    this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    if (this->queue == NULL) {
        // Revert to CPU if GPU is unavailable
        this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
}

Network::Network(std::string location) {
    // Open Network Save
    std::ifstream model_struct;
    std::string model_location = location;
    model_struct.open(model_location.append("model.struct"));
    
    this->g = new Graph();
    
    this->input = new Tensor(NetworkBufferParse::LoadInput(&model_struct));
    
    // Initialise OpenCL dispatch queue
    this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    if (this->queue == NULL) {
        // Revert to CPU if GPU is unavailable
        this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    
    std::string line;
    GraphLoader::LayerDetails layer;
    
    // Load First Layer
    getline(model_struct, line);
    
    g->InsertInput(this->input);
    
    // While the next Layer is a Layer
    while (GraphLoader::ParseLayer(line, &layer)) {
        g->InsertLayer(location, layer.type, layer.id, layer.inputs, layer.dependents);
        // Load next line
        getline(model_struct, line);
    }
    
    this->g->InitialiseLayers(this->input);
    
    this->g->LoadLayers(location);
    
    this->prediction = new Tensor(this->g->GetOutputSize());
    this->output = new Tensor(this->g->GetOutputSize());
    
    // Implement Data Loading Methods
    std::string input_pipeline_location = location;
    std::string output_pipeline_location = location;
    input_pipeline_location.append("input_pipeline.instr");
    output_pipeline_location.append("output_pipeline.instr");
    
    this->input_pipeline = new InstructionInterpreter(input_pipeline_location, this->input);
    this->output_pipeline = new InstructionInterpreter(output_pipeline_location, this->prediction);
    
    this->g->AddLoss(GraphLoader::ParseLoss(line));
    
    model_struct.close();
}

void Network::Evaluate() {
    this->input_pipeline->LoadNextDataBatch();
    this->output_pipeline->LoadNextDataBatch();
    
    this->g->Evaluate(this->input, this->output, &(this->queue));
}

float Network::Learn() {
    this->input_pipeline->LoadNextDataBatch();
    this->output_pipeline->LoadNextDataBatch();
    
    return this->g->Learn(this->input, this->prediction, &(this->queue), this->LearningRate);
}

bool Network::SaveNetwork(std::string loc) {
    this->g->Save(loc, this->input->dims, this->prediction->dims);
    return true;
}
