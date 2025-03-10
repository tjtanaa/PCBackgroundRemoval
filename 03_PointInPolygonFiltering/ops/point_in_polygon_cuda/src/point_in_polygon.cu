__global__ void pipnumberSinglePolygonKernel(
    const float* __restrict__ points,
    const float* __restrict__ polygon_vertices,
    const int num_points,
    const int num_vertices,
    int * __restrict__ wn_results
){
    // points (N, 3)
    // vertices (N +1, 2), the last is the duplicate of the first
    // wc_results (N)

    int t_id = blockIdx.x * blockDim.x + threadIdx.x; 
    // partition threads to single (point_i, edge_j)
    for (; t_id < num_points * (num_vertices - 1); t_id += (blockDim.x * gridDim.x) ){
        int point_i = t_id / (num_vertices - 1);
        int edge_j = t_id % (num_vertices - 1);

        if( (point_i < num_points) & (edge_j < num_vertices - 1) ){
            if(polygon_vertices[edge_j * 2 + 1] <= points[point_i *3 + 1]){
                if(polygon_vertices[(edge_j+1)* 2 + 1] > points[point_i *3 + 1]){
                    if( 
                        ( 
                            (polygon_vertices[(edge_j+1)* 2 +0] - polygon_vertices[edge_j * 2 + 0] ) * (points[point_i *3 +1]  - polygon_vertices[edge_j * 2 + 1]) - (points[point_i * 3 + 0]  -  polygon_vertices[edge_j * 2 + 0] ) * (polygon_vertices[(edge_j+1)* 2 +1] - polygon_vertices[edge_j * 2 + 1]) 
                        ) > 0
                    ){
                        atomicAdd(&wn_results[point_i], 1);
                    }
                }
            } 
            else{
                if(polygon_vertices[(edge_j+1)* 2 +1] <= points[point_i *3 +1]){
                    if( 
                        ( 
                            (polygon_vertices[(edge_j+1)* 2 +0] - polygon_vertices[edge_j * 2 + 0] ) * (points[point_i *3 +1]  - polygon_vertices[edge_j * 2 + 1]) - (points[point_i *3 +0]  -  polygon_vertices[edge_j * 2 + 0] )  * (polygon_vertices[(edge_j+1)* 2 +1] - polygon_vertices[edge_j * 2 + 1]) 
                        ) < 0
                    ){
                        atomicAdd(&wn_results[point_i], -1);
                    }
                }
            }
        }

    }

}


void pipnumberSinglePolygonLauncher(
    const float*  points,
    const float*  polygon_vertices,
    const int num_points,
    const int num_vertices,
    int *  wn_results){

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pipnumberSinglePolygonKernel, 0, num_points *(num_vertices - 1));
    gridSize = (num_points * (num_vertices - 1) + blockSize - 1) / blockSize;

    pipnumberSinglePolygonKernel<<<gridSize,blockSize>>>(points, polygon_vertices, num_points, num_vertices, wn_results);
}
    