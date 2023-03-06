#include <onnx.h>


static const char* test_base_dir = "/home/lzy/Code/C/onnx-learn/test_data/";

const char* join(const char* s1, const char* s2){
  char* res = malloc(strlen(s1)+strlen(s2)+1);
  memset(res,0,strlen(s1)+strlen(s2)+1);
  sprintf(res, "%s%s", s1,s2);
  return res;
}

void test_split(
    const char* model_path
    , const char* data_path
    , const char* split_path
    , const char** output_path
    , int n_output
    ){
  struct onnx_context_t *ctx;
  struct onnx_tensor_t *data;
  struct onnx_tensor_t *split;


  struct onnx_tensor_t *input_data;
  struct onnx_tensor_t *input_split;

  struct onnx_tensor_t *output;

  printf("---------\n");
  ctx = onnx_context_alloc_from_file(model_path, NULL, 0);
  printf("alloc ctx %d\n", ctx!=NULL);
  printf("data_path %s\n",data_path);
  data = onnx_tensor_alloc_from_file(data_path);
  printf("alloc data %d\n", data!=NULL);
  printf("alloc data  show %f\n", ((float*)data->datas)[0]);
  split = onnx_tensor_alloc_from_file(split_path);
  printf("alloc split %d\n", split!=NULL);

  printf("=====================\n");
  input_data = onnx_tensor_search(ctx,"input");
  input_split = onnx_tensor_search(ctx,"split");


  printf("=====================\n");

  if(input_data){
    onnx_tensor_apply(input_data, data->datas, data->ndata*onnx_tensor_type_sizeof(data->type));
    onnx_tensor_free(data);
  }

  if(input_split){
    onnx_tensor_apply(input_split, split->datas, split->ndata*onnx_tensor_type_sizeof(split->type));
    onnx_tensor_free(split);
  }
  printf("=====================\n");
  onnx_run(ctx);
  printf("=====================\n");

  for(int i = 0; i < n_output; ++i){
    char name[256];
    memset(name,0,256);
    sprintf(name,"output_%d",i+1);
    output = onnx_tensor_search(ctx,name);
    struct onnx_tensor_t *gt = onnx_tensor_alloc_from_file(output_path[i]);
    int b = onnx_tensor_equal(gt,output);
    printf("idx %d same %d\n",i,b);
  }

}

int main(int argc, char** argv){
#if 1
  const char * paths[] = {
    join(test_base_dir, "test_split_equal_parts_1d/test_data_set_0/output_0.pb"),
    join(test_base_dir, "test_split_equal_parts_1d/test_data_set_0/output_1.pb"),
    join(test_base_dir, "test_split_equal_parts_1d/test_data_set_0/output_2.pb"),
  };
  test_split(
      join(test_base_dir,"test_split_equal_parts_1d/model.onnx")
      , join(test_base_dir, "test_split_equal_parts_1d/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_split_equal_parts_1d/test_data_set_0/input_1.pb")
      , paths,
      3
      );
#endif

#if 1
  const char * paths1[] = {
    join(test_base_dir, "test_split_equal_parts_2d/test_data_set_0/output_0.pb"),
    join(test_base_dir, "test_split_equal_parts_2d/test_data_set_0/output_1.pb"),
  };
  test_split(
      join(test_base_dir,"test_split_equal_parts_2d/model.onnx")
      , join(test_base_dir, "test_split_equal_parts_2d/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_split_equal_parts_2d/test_data_set_0/input_1.pb")
      , paths1,
      2
      );
#endif


#if 1
  const char * paths2[] = {
    join(test_base_dir, "test_split_equal_parts_default_axis/test_data_set_0/output_0.pb"),
    join(test_base_dir, "test_split_equal_parts_default_axis/test_data_set_0/output_1.pb"),
    join(test_base_dir, "test_split_equal_parts_default_axis/test_data_set_0/output_2.pb"),
  };
  test_split(
      join(test_base_dir,"test_split_equal_parts_default_axis/model.onnx")
      , join(test_base_dir, "test_split_equal_parts_default_axis/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_split_equal_parts_default_axis/test_data_set_0/input_1.pb")
      , paths2,
      3
      );
#endif

#if 1
  const char * paths3[] = {
    join(test_base_dir, "test_split_variable_parts_1d/test_data_set_0/output_0.pb"),
    join(test_base_dir, "test_split_variable_parts_1d/test_data_set_0/output_1.pb"),
  };
  test_split(
      join(test_base_dir,"test_split_variable_parts_1d/model.onnx")
      , join(test_base_dir, "test_split_variable_parts_1d/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_split_variable_parts_1d/test_data_set_0/input_1.pb")
      , paths3,
      2
      );
#endif

#if 1
  const char * paths4[] = {
    join(test_base_dir, "test_split_variable_parts_2d/test_data_set_0/output_0.pb"),
    join(test_base_dir, "test_split_variable_parts_2d/test_data_set_0/output_1.pb"),
  };
  test_split(
      join(test_base_dir,"test_split_variable_parts_2d/model.onnx")
      , join(test_base_dir, "test_split_variable_parts_2d/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_split_variable_parts_2d/test_data_set_0/input_1.pb")
      , paths4,
      2
      );
#endif

#if 1
  const char * paths5[] = {
    join(test_base_dir, "test_split_variable_parts_default_axis/test_data_set_0/output_0.pb"),
    join(test_base_dir, "test_split_variable_parts_default_axis/test_data_set_0/output_1.pb"),
  };
  test_split(
      join(test_base_dir,"test_split_variable_parts_default_axis/model.onnx")
      , join(test_base_dir, "test_split_variable_parts_default_axis/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_split_variable_parts_default_axis/test_data_set_0/input_1.pb")
      , paths5,
      2
      );
#endif
  return 0;
}
