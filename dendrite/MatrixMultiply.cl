#define INDEX2D(x, y, width) x + width * y

kernel void MatrixMultiply(global float* i, int i_w, global float* o, global float* w) {
    size_t node_index = get_global_id(0);
    
    float sum = 0;
    
    for (size_t loc = 0; loc < i_w; loc++) {
        sum += i[loc] * w[INDEX2D(loc, node_index, i_w)];
    }
    
    o[node_index] = sum;
}

kernel void MatrixMultiplyTranspose(global float* i, int i_w, global float*o, global float* w) {
    size_t output_index = get_global_id(0);
    size_t output_size = get_global_size(0);
    
    float sum = 0;
    
    for (size_t loc = 0; loc < i_w; loc++) {
        sum += i[loc] * w[INDEX2D(output_index, loc, output_size)];
    }
    
    o[output_index] = sum;
}

kernel void CalculateWeightDerivatives(global float* er, global float* inp, global float* weight_derivative) {
    size_t i = get_global_id(0);
    size_t w = get_global_size(0);
    size_t j = get_global_id(1);
    
    weight_derivative[INDEX2D(i, j, w)] = inp[i] * er[j];
}

kernel void UpdateWeights(global float* deriv, global float* wei, float eta) {
    size_t weight_index = get_global_id(0);
    if (!isnan(deriv[weight_index])) {
        wei[weight_index] = wei[weight_index] - eta * deriv[weight_index];
    }
}
