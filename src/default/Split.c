#include <onnx.h>


struct operator_pdata_t {
	int axis;
	int split[];
};


static int Split_init_op1(struct onnx_node_t * n){
	struct operator_pdata_t *pdat;
	struct onnx_tensor_t* split = NULL;
	printf("split ninput %d\n",n->ninput);
	printf("opset %d\n",n->opset);
	int64_t* splits;
	if(n->ninput>=1 && n->noutput>=1){
		if(n->ninput>1){
			split = n->inputs[1];
		}
		if(split){
			pdat = malloc(sizeof(struct operator_pdata_t) + sizeof(int) * split->ndata);
			if(pdat){
				pdat->axis = onnx_attribute_read_int(n,"axis",0);
				for(int i = 0; i < split->ndata;++i){
					pdat->split[i] = ((int*)split->datas)[i];
				}
				n->priv = pdat;
				return 1;
			}
		}else{
			int n_split = onnx_attribute_read_ints(n, "split", &splits);
			if(n_split>0){
				pdat = malloc(sizeof(struct operator_pdata_t) + sizeof(int)*n_split);
				if(pdat){
					pdat->axis = onnx_attribute_read_int(n,"axis",0);
					for(int i = 0; i < n_split; ++i){
						pdat->split[i] = splits[i];
					}
					n->priv = pdat;
					return 1;
				}
			}else{
				pdat = malloc(sizeof(struct operator_pdata_t) + sizeof(int)*n->noutput);
				if(pdat){
					pdat->axis = onnx_attribute_read_int(n,"axis",0);
					int v_dims = n->inputs[0]->dims[pdat->axis];
					for(int i = 0; i < n->noutput; ++i){
						pdat->split[i] = v_dims / n->noutput;
					}
					n->priv = pdat;
					return 1;
				}
			}
		}
	}
	return 0;
}

static int Split_init_op11(struct onnx_node_t * n){
	struct operator_pdata_t *pdat;
	struct onnx_tensor_t* split = NULL;
	if(n->ninput>=1 && n->noutput>=1){
		if(n->ninput>1){
			split = n->inputs[1];
		}
		if(split){
			assert(split->ndim==1);
			assert(split->ndata==n->noutput);
			pdat = malloc(sizeof(struct operator_pdata_t) + sizeof(int) * split->ndata);
			if(pdat){
				pdat->axis = onnx_attribute_read_int(n,"axis",0);
				for(int i = 0; i < split->ndata;++i){
					pdat->split[i] = ((int*)split->datas)[i];
				}
				n->priv = pdat;
				return 1;
			}
		}else{
			pdat = malloc(sizeof(struct operator_pdata_t) + sizeof(int)*n->noutput);
			if(pdat){
				pdat->axis = onnx_attribute_read_int(n,"axis",0);
				int v_dims = n->inputs[0]->dims[pdat->axis];
				// printf("v_dims %d\n",v_dims);
				// printf("ndata %ld\n",split->ndata);
				for(int i = 0; i < n->noutput; ++i){
					pdat->split[i] = v_dims / n->noutput;
				}
				n->priv = pdat;
				printf("hhhh ======== %d\n",v_dims);
				return 1;
			}
		}
	}
	return 0;
}

static int Split_exit(struct onnx_node_t * n){
	struct operator_pdata_t *pdat = (struct operator_pdata_t*)n->priv;
	if(pdat){
		free(pdat);
	}
	return 1;
}
static int Split_reshape(struct onnx_node_t * n){
	// printf("split reshape begin\n");
	struct operator_pdata_t *pdat = (struct operator_pdata_t*)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * o;
	int sum_axis = 0;
	int dims[x->ndim];
	for(int i = 0; i < n->noutput; ++i){
		sum_axis += pdat->split[i];
	}
	if(sum_axis != n->inputs[0]->dims[pdat->axis]){
		return 0;
	}
	for(int i = 0; i < n->noutput; ++i){
		o = n->outputs[i];
		for(int j = 0; j < x->ndim; ++j){
			dims[j] = x->dims[j];
		}
		dims[pdat->axis] = pdat->split[i];
		if(!onnx_tensor_reshape(o, dims, x->ndim, x->type)){
			return 0;
		}
	}
	// printf("split reshape end\n");
	return 1;
}

static void Split_operator(struct onnx_node_t* n){
	struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
	struct onnx_tensor_t * x = n->inputs[0];
	struct onnx_tensor_t * o;
	//TODO assert type is not string
	assert(x->type != ONNX_TENSOR_TYPE_STRING);
	int sz = onnx_tensor_type_sizeof(x->type);
	int axis = pdat->axis;

	char* px = (char*)x->datas;
	char* py;
	int sum_before_axis = 0;
	int mul_before_axis = 1;
	for(int i = 0; i < axis; ++i){
		sum_before_axis += x->strides[i] * x->dims[i];
	}
	for(int i = 0; i < axis; ++i){
		mul_before_axis *= x->dims[i];
	}
	int sum_axis = 0;
	int begin_x;
	int begin_y = 0;
	for(int io = 0; io < n->noutput; ++io){
		// printf("io %d\n",pdat->split[io]);
		o = n->outputs[io];
		py = (char*)o->datas;
		// printf("begin_x %d\n",begin_x);
		begin_y = 0;
		for(int j = 0; j < mul_before_axis; ++j){
			for(int i = sum_axis; i < sum_axis+pdat->split[io]; i++){
				begin_x = j * (axis >= 1?x->strides[axis-1]:0)+ i * x->strides[axis];
				memcpy(py+begin_y*sz,px+begin_x*sz, x->strides[axis] * sz);

				begin_y+=o->strides[axis];
			}
		}
		sum_axis += pdat->split[io];
	}
}

void resolver_default_op_Split(struct onnx_node_t * n)
{
	if(n->opset >= 18){
		//TODO opset18 is not implemented. there has a num_output attribute
		n->init = NULL;
		n->exit = Split_exit;
		n->reshape = Split_reshape;
		n->operator = Split_operator;
	}
	else if(n->opset >= 13)
	{
		n->init = Split_init_op11;
		n->exit = Split_exit;
		n->reshape = Split_reshape;
		n->operator = Split_operator;
	}
	else if(n->opset >= 11)
	{
		n->init = Split_init_op11;
		n->exit = Split_exit;
		n->reshape = Split_reshape;
		n->operator = Split_operator;
	}
	else if(n->opset >= 2)
	{
		n->init = Split_init_op1;
		n->exit = Split_exit;
		n->reshape = Split_reshape;
		n->operator = Split_operator;
	}
	else if(n->opset >= 1)
	{
		n->init = Split_init_op1;
		n->exit = Split_exit;
		n->reshape = Split_reshape;
		n->operator = Split_operator;
	}
}
