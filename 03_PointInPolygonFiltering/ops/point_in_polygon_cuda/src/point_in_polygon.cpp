/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include <cuda.h>

using namespace tensorflow;

REGISTER_OP("PipnumberSinglePolygon")
  .Input("points: float32")
  .Input("polygon_vertices: float32")
  .Output("wn_results: int32");


void pipnumberSinglePolygonLauncher(
    const float*  points,
    const float*  polygon_vertices,
    const int num_points,
    const int num_vertices,
    int *  wn_results);

class PipnumberSinglePolygonGpuOp: public OpKernel{
  public:
    explicit PipnumberSinglePolygonGpuOp(OpKernelConstruction* context):OpKernel(context){}
    void Compute(OpKernelContext* context) override {

        const Tensor& points = context->input(0);
        auto points_ptr = points.template flat<float>().data();
        OP_REQUIRES(context, points.dims()==2 && points.dim_size(1)==3,
                    errors::InvalidArgument("PipnumberSinglePolygonGpuOp expects points in shape: [input_point_nums, 3]."));

        const Tensor& polygon_vertices = context->input(1);
        auto polygon_vertices_ptr = polygon_vertices.template flat<float>().data();
        OP_REQUIRES(context, polygon_vertices.dims()==2 && polygon_vertices.dim_size(1)==2,
                    errors::InvalidArgument("PipnumberSinglePolygonGpuOp expects polygon_vertices in shape: [input_point_nums, 2]."));

        int num_points = points.dim_size(0);
        int num_vertices = polygon_vertices.dim_size(0);

        Tensor* wn_results = nullptr;
        auto wn_results_shape = TensorShape({num_points});
        OP_REQUIRES_OK(context, context->allocate_output(0, wn_results_shape, &wn_results));
        int* wn_results_ptr = wn_results->template flat<int>().data();
        cudaMemset(wn_results_ptr, 0, num_points*sizeof(int));
//        printf("RoI Pooling Feature Size: %d MB\n", roi_num*voxel_num*channels*sizeof(float)/1024/1024);
//        cudaMemset(output_features_ptr, padding_value, kernel_number*voxel_num*channels*sizeof(float));
        pipnumberSinglePolygonLauncher(
            points_ptr,
            polygon_vertices_ptr,
            num_points,
            num_vertices,
            wn_results_ptr);
    };
};
REGISTER_KERNEL_BUILDER(Name("PipnumberSinglePolygon").Device(DEVICE_GPU), PipnumberSinglePolygonGpuOp);