#include <onnx.h>

static int Resize_init(struct onnx_node_t * n){
	return 1;
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
