kernel void ElementWiseAdd(global float* i, global float* o, global float* delta) {
    size_t pos = get_global_id(0);
    size_t delta_pos = get_local_id(0);
    
    o[pos] = i[pos] + delta[delta_pos];
}
