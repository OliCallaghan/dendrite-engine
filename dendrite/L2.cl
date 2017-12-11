kernel void L2_Val_Reduce(global float* output, size_t o_l, global float* prediction, global float* intermediate) {
    float sum = 0;
    for (int pos = 0; pos < o_l; pos++) {
        sum += 0.5 * pown(output[pos] - prediction[pos], 2);
    }
    intermediate[0] = sum;
}

kernel void L2(global float* output, global float* prediction, global float* deltamap) {
    size_t pos = get_global_id(0);
    deltamap[pos] = output[pos] - prediction[pos];
}
