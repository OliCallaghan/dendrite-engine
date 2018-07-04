kernel void Tangent(global float* input, global float* output) {
    size_t pos = get_global_id(0);
    output[pos] = tanh(input[pos]);
}

kernel void Backprop_Tangent(global float* input, global float* delta, global float* backprop) {
    size_t pos = get_global_id(0);
    float cos_tmp = cosh(input[pos]);
    backprop[pos] = delta[pos] / (cos_tmp * cos_tmp);
    backprop[pos] = delta[pos];
}
