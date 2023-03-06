#include <limits.h>
#include <onnx.h>
#include <stdio.h>
#include <string.h>
#include <marcos.h>

struct slice_pdata_t {
	int *starts; 
	int n_start;
	int *ends; 
	int n_end;
	int *steps; 
	int n_step;
	int *axes; 
	int n_axes;
};

static int Slice_init(struct onnx_node_t *n){
	if((n->ninput>=3) && (n->noutput==1)){
		return 1;
	}
	return 0;
}

static int Slice_exit(struct onnx_node_t *n){
	return 1;
}

static int Slice_reshape(struct onnx_node_t *n){
	return 1;
}

int slice_float32_recursive(
		int* src_shape,
		int n_src_shape,
		int* dst_shape,
		int n_dst_shape,
		float* src,
		const int64_t *starts,
		const int64_t *steps,
		const int64_t *axes,
		const int64_t *stride_in,
		const int64_t *stride_out,
		const int64_t dim_dix,
		const int64_t n_axes,
		float *dst
		){
	// int64_t output_length = dst_shape[axes[dim_dix]]*stride_out[axes[dim_dix]];
	int64_t output_length = dst_shape[dim_dix];
	// printf("output len %ld\n",output_length);
	// printf("nsrc len %d\n",n_src_shape);
	if(dim_dix == n_src_shape - 1){
		for(int64_t i = 0; i < output_length; ++i){
			int64_t src_i = starts[dim_dix] + i * steps[dim_dix];
			// printf("dim idx  %ld\n",dim_dix);
			// printf("src_i %ld\n",src_i);
			// printf("steps %ld\n",steps[dim_dix]);
			// printf("strides in %ld\n",stride_in[dim_dix]);
			// printf("stride  %ld\n",stride_in[dim_dix]);
			dst[i] = src[src_i];
		}
	}else{
		for(int64_t i = 0; i < output_length; ++i){
			int64_t src_i = starts[dim_dix] + i * steps[dim_dix];
			// printf("dim idx  %ld\n",dim_dix);
			// printf("rec  src_i %ld\n",src_i);
			// printf("steps %ld\n",steps[dim_dix]);
			// printf("strides in %ld\n",stride_in[dim_dix]);
			slice_float32_recursive(src_shape,n_src_shape,dst_shape,n_dst_shape,
					src + src_i * stride_in[dim_dix],
					starts,
					steps,
					axes,
					stride_in,
					stride_out,
					dim_dix+1,
					n_axes,
					dst + i*stride_out[dim_dix]
					);
		}
	}
	return 1;
}

static void Slice_float32(struct onnx_node_t *n){
	struct onnx_tensor_t *y = n->outputs[0];
	struct onnx_tensor_t *data = n->inputs[0];
	struct onnx_tensor_t *starts = n->inputs[1];
	struct onnx_tensor_t *ends = n->inputs[2];
	struct onnx_tensor_t *axes = NULL;
	//n->inputs[3];
	struct onnx_tensor_t *steps = NULL;
	int n_axes = starts->ndim;
	assert(n_axes == starts->ndim);
	assert(n_axes == 1);
	// printf("xxxx %d\n",n_axes);
	int64_t pstarts[TENSOR_MAX_NDIM] = {0};
	int64_t pends[TENSOR_MAX_NDIM] = {0};
	int64_t paxes[TENSOR_MAX_NDIM] = {0};
	int64_t psteps[TENSOR_MAX_NDIM] = {1};
	int64_t stride_in[data->ndim];
	int64_t stride_out[y->ndim];

	for(int i = 0; i < data->ndim; ++i){
		stride_in[i] = data->strides[i];
		stride_out[i] = y->strides[i];
	}
	if(n->ninput > 3){
		axes = n->inputs[3];
	}
	if(n->ninput > 4){
		steps = n->inputs[4];
	}
	for(int i = 0; i < data->ndim; ++i){
		pstarts[i] = 0;
		psteps[i] = 1;
		pends[i] = data->dims[i];
		paxes[i] = i;
	}
	for(int i = 0; i < starts->dims[0]; ++i){
		int64_t s = ((int64_t*)starts->datas)[i];
		int64_t e = ((int64_t*)ends->datas)[i];
		int64_t a = i;
		int64_t step = 1;
		if(axes){
			a = ((int64_t*)axes->datas)[i];
		}
		if(steps){
			step = ((int64_t*)steps->datas)[i];
		}
		if(a < 0){
			a = a + data->ndim;
		}
		assert(a>=0 && a<data->ndim);
		if(s == LONG_MIN){
			s = 0;
		}
		if(s == LONG_MAX || s >= data->dims[a]){
			s = data->dims[a]-1;
		}
		if(s<0){
			s = s + data->dims[a];
		}
		if(e == LONG_MAX || e > data->dims[a]){
			e = data->dims[a];
		}
		if(e<0){
			if(-e > data->dims[a]){
				e = -1;
			}else{
				e = e + data->dims[a];
			}
		}

		pstarts[a] = s;
		pends[a] = e;
		psteps[a] = step;
	}
	// printf("n_shape %d\n",data->ndim);
	// printf("start %ld\n",pstarts[0]);
	// printf("end %ld\n",pends[0]);
	// printf("steps %ld\n",psteps[0]);
	// printf("axes %ld\n",paxes[0]);
	// printf("data shape %d %d %d %d\n"
	// 		,data->dims[0]
	// 		,data->dims[1]
	// 		,data->dims[2]
	// 		,data->dims[3]
	// 		);
	// printf("data strides %ld %ld %ld %ld\n"
	// 		,stride_in[0]
	// 		,stride_in[1]
	// 		,stride_in[2]
	// 		,stride_in[3]
	// 		);
	// printf("y shape %d %d %d %d\n"
	// 		,y->dims[0]
	// 		,y->dims[1]
	// 		,y->dims[2]
	// 		,y->dims[3]
	// 		);
	// printf("y strides %ld %ld %ld %ld\n"
	// 		,stride_out[0]
	// 		,stride_out[1]
	// 		,stride_out[2]
	// 		,stride_out[3]
	// 		);
	// for(int i = 0; i < data->ndim; ++i){
	// 	printf("start %ld\n",pstarts[i]);
	// 	printf("end %ld\n",pends[i]);
	// }
	slice_float32_recursive(data->dims, data->ndim, y->dims, y->ndim
			, (float*)data->datas
			, pstarts
			, psteps
			, paxes
			, stride_in
			, stride_out
			, 0
			, n_axes
			, (float*)y->datas
			);
}
void resolver_default_op_Slice(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		n->init = Slice_init;
		n->exit = Slice_exit;
		n->reshape = Slice_reshape;
		n->operator = Slice_float32;
	}
	else if(n->opset >= 11)
	{
		n->init = Slice_init;
		n->exit = Slice_exit;
		n->reshape = Slice_reshape;
		n->operator = Slice_float32;
	}
	else if(n->opset >= 10)
	{
		n->init = Slice_init;
		n->exit = Slice_exit;
		n->reshape = Slice_reshape;
		n->operator = Slice_float32;
	}
	else if(n->opset >= 1)
	{
		n->init = Slice_init;
		n->exit = Slice_exit;
		n->reshape = Slice_reshape;
		n->operator = Slice_float32;
	}
}
