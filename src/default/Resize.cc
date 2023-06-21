#include <onnx.h>

// FIXME Do not implement some characters in tensorflow, such as crop_and_resize
struct operator_pdata_t {
	char coordinate_transformation_mode[24];
	float cubic_coeff_a;
	int exclude_outside;
	float extrapolation_value;
	char mode[12];
	char nearest_mode[24];
};

static int Resize_init_11(struct onnx_node_t * n){
	struct operator_pdata_t * pdat;
	if(n->ninput>=3 && n->noutput ==1){
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat){
			memset(pdat,0,sizeof(struct operator_pdata_t));
			const char* ctm = onnx_attribute_read_string(n,"coordinate_transformation_mode","half_pixel");
			memcpy(pdat->coordinate_transformation_mode,ctm,strlen(ctm));
			pdat->cubic_coeff_a = onnx_attribute_read_float(n,"cubic_coeff_a",-0.75);
			pdat->exclude_outside = onnx_attribute_read_int(n,"exclude_outside",0);
			pdat->extrapolation_value = onnx_attribute_read_float(n,"extrapolation_value",0.0);
			const char* cmode = onnx_attribute_read_string(n,"mode","nearest"); 
			memcpy(pdat->mode,cmode,strlen(cmode));
			const char* cnearest_mode = onnx_attribute_read_string(n,"nearest_mode","round_prefer_floor"); 
			memcpy(pdat->nearest_mode,cnearest_mode,strlen(cnearest_mode));
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}


static int Resize_exit_11(struct onnx_node_t * n){
	struct operator_pdata_t* pdat = (struct operator_pdata_t*)n->priv;
	if(pdat){
		free(pdat);
	}
	return 1;
}

static int Resize_reshape(struct onnx_node_t * n){
	struct onnx_tensor_t* x  = n->inputs[0];
	struct onnx_tensor_t* scale  = n->inputs[1];
	struct onnx_tensor_t* y  = n->outputs[0];
	int rank = scale->ndata;
	float scales[rank];
	for(int i = 0; i<rank; ++i){
		scales[i] = ((float*)scale->datas)[i];
	}
	int output_size[rank];
	for(int i = 0; i < rank; ++i){
		output_size[i] = (int)(x->dims[i] * scales[i]);
	}
	return onnx_tensor_reshape(y,output_size,x->ndim,x->type);
}

int resize2d_linear_fp32(
		struct onnx_tensor_t* x,
		struct onnx_tensor_t* y,
		const float scale_h,
		const float scale_w
		){
	int* src_shape = x->dims;
	int* dst_shape = y->dims;
	assert(x->ndata==4);
	int num_imgs = src_shape[0]*src_shape[1];
	int src_h = src_shape[2];
	int src_w = src_shape[3];
	int dst_h = dst_shape[2];
	int dst_w = dst_shape[3];

	const float hscale = 1.0f / scale_h;
	const float wscale = 1.0f / scale_w;
	float* src = (float*)x->datas;
	float* dst = (float*)y->datas;
	for(int i = 0; i < num_imgs; ++i){
		const float* lsrc = src + i*src_h*src_w;
		float* ldst = dst + i*dst_h*dst_w;
		for(int64_t oh = 0; oh < dst_h; ++oh){
			float ih = dst_h>1 ? (oh+0.5f)*hscale-0.5f : 0;
			int64_t h0, h1;
			float h0_lambda, h1_lambda;
			if(ih < 0){
				h0 = 0;
				h1 = 0;
				h0_lambda = 1;
				h1_lambda = 0;
			}else{
				h0 = (int64_t)ih;
				h1 = h0 + (h0<src_h-1);
				h1_lambda = ih-h0;
				h0_lambda = 1.0f - h1_lambda;
			}
			for(int64_t ow = 0; ow < dst_w; ++ow){
				float iw = dst_w>1 ? (ow+0.5f)*wscale-0.5f : 0;
				int64_t w0, w1;
				float w0_lambda, w1_lambda;
				if(iw < 0){
					w0 = 0;
					w1 = 0;
					w0_lambda = 1;
					w1_lambda = 0;
				}else{
					w0 = (int64_t)iw;
					w1 = w0 + (w0<src_w-1);
					w1_lambda = iw-w0;
					w0_lambda = 1.0f - w1_lambda;
				}

				ldst[oh*dst_w+ow] = lsrc[h0*src_w+w0] * h0_lambda*w0_lambda+
														lsrc[h0*src_w+w1] * h0_lambda*w1_lambda+
														lsrc[h1*src_w+w0] * h1_lambda*w0_lambda+
														lsrc[h1*src_w+w1] * h1_lambda*w1_lambda;
			}
		}
	}
	return 1;
}

static int Resize_op_11(struct onnx_node_t * n){
	struct operator_pdata_t* priv = (struct operator_pdata_t*)n->priv;
	struct onnx_tensor_t* x  = n->inputs[0];
	struct onnx_tensor_t* scale  = n->inputs[1];
	struct onnx_tensor_t* y  = n->outputs[0];
	char* mode = priv->mode;
	if(scale->ndata==4){
		float* scales = (float*)scale->ndata;
		assert(fabs(scales[0]-1)<1e-5 && fabs(scales[1]-1)<1e-5);
	}
	if(strcmp(mode,"nearest")==0){
	}
}
void resolver_default_op_Resize(struct onnx_node_t * n)
{
		// switch (n->inputs[0]->type) {
		// case ONNX_TENSOR_TYPE_BOOL:
		// case ONNX_TENSOR_TYPE_INT8:
		// case ONNX_TENSOR_TYPE_INT16:
		// case ONNX_TENSOR_TYPE_INT32:
		// case ONNX_TENSOR_TYPE_INT64:
		// case ONNX_TENSOR_TYPE_UINT8:
		// case ONNX_TENSOR_TYPE_UINT16:
		// case ONNX_TENSOR_TYPE_UINT32:
		// case ONNX_TENSOR_TYPE_UINT64:
		// case ONNX_TENSOR_TYPE_BFLOAT16:
		// case ONNX_TENSOR_TYPE_FLOAT16:
		// case ONNX_TENSOR_TYPE_FLOAT32:
		// case ONNX_TENSOR_TYPE_FLOAT64:
		// case ONNX_TENSOR_TYPE_COMPLEX64:
		// case ONNX_TENSOR_TYPE_COMPLEX128:
		// case ONNX_TENSOR_TYPE_STRING:
		// 	n->init = Resize_init;
		// 	n->exit = Resize_exit;
		// 	n->reshape = Resize_reshape;
		// 	n->operator = Resize_operator;
		// 	break;
		// default:
		// 	break;
		// }
	if(n->opset >= 13)
	{
	}
	else if(n->opset >= 11)
	{
	}
	else if(n->opset >= 10)
	{
	}
}
