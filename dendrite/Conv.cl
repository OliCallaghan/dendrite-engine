#define FLATTEN4D(x,y,z,n,width,height,depth) (x + y*width + z * width * height + n * width * height * depth)

kernel void Convolve(global float* input, global float* kern, global float* output, int kernel_x, int kernel_y, int kernel_z, int kernel_n) {
    size_t pos_x = get_global_id(0);
    size_t pos_y = get_global_id(1);
    size_t pos_z = get_global_id(2);
    
    int x;
    int y;
    int z;
    int n;
    
    float sum = 0;
    
    for (x = 0; x < kernel_x; x++) {
        for (y = 0; y < kernel_y; y++) {
            for (z = 0; z < kernel_z; z++) {
                for (n = 0; n < kernel_n; n++) {
                    //sum += input[
                }
            }
        }
    }
}
