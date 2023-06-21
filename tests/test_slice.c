#include <onnx.h>


static const char* test_base_dir = "/home/lzy/Code/C/libonnx-F/test_data/";

const char* join(const char* s1, const char* s2){
  char* res = malloc(strlen(s1)+strlen(s2)+1);
  memset(res,0,strlen(s1)+strlen(s2)+1);
  sprintf(res, "%s%s", s1,s2);
  return res;
}

void test_slice(
    const char* model_path
    , const char* data_path
    , const char* starts_path
    , const char* ends_path
    , const char* axes_path
    , const char* steps_path
    , const char* output_path
    ){
  struct onnx_context_t *ctx;
  struct onnx_tensor_t *data;
  struct onnx_tensor_t *starts;
  struct onnx_tensor_t *ends;
  struct onnx_tensor_t *axes;
  struct onnx_tensor_t *steps;


  struct onnx_tensor_t *input_data;
  struct onnx_tensor_t *input_starts;
  struct onnx_tensor_t *input_ends;
  struct onnx_tensor_t *input_axes;
  struct onnx_tensor_t *input_steps;

  struct onnx_tensor_t *output;

  ctx = onnx_context_alloc_from_file(model_path, NULL, 0);
  printf("alloc ctx %d\n", ctx!=NULL);
  data = onnx_tensor_alloc_from_file(data_path);
  printf("alloc data %d\n", data!=NULL);
  starts = onnx_tensor_alloc_from_file(starts_path);
  printf("alloc starts %d\n", starts!=NULL);
  ends = onnx_tensor_alloc_from_file(ends_path);
  printf("alloc ends %d\n", ends!=NULL);
  axes = onnx_tensor_alloc_from_file(axes_path);
  printf("alloc axes %d\n", axes!=NULL);
  steps = onnx_tensor_alloc_from_file(steps_path);
  printf("alloc steps %d\n", steps!=NULL);

  printf("=====================\n");
  input_data = onnx_tensor_search(ctx,"x");
  input_starts = onnx_tensor_search(ctx,"starts");
  input_ends = onnx_tensor_search(ctx,"ends");
  input_axes = onnx_tensor_search(ctx,"axes");
  input_steps = onnx_tensor_search(ctx,"steps");

  output = onnx_tensor_search(ctx,"y");

  printf("=====================\n");

  if(input_data){
    onnx_tensor_apply(input_data, data->datas, data->ndata*onnx_tensor_type_sizeof(data->type));
    onnx_tensor_free(data);
  }

  if(input_starts){
    onnx_tensor_apply(input_starts, starts->datas, starts->ndata*onnx_tensor_type_sizeof(starts->type));
    onnx_tensor_free(starts);
  }

  if(input_ends){
    onnx_tensor_apply(input_ends, ends->datas, ends->ndata*onnx_tensor_type_sizeof(ends->type));
    onnx_tensor_free(ends);
  }

  if(input_axes){
    onnx_tensor_apply(input_axes, axes->datas, axes->ndata*onnx_tensor_type_sizeof(axes->type));
    onnx_tensor_free(axes);
  }

  if(input_steps){
    onnx_tensor_apply(input_steps, steps->datas, steps->ndata*onnx_tensor_type_sizeof(steps->type));
    onnx_tensor_free(steps);
  }

  struct onnx_tensor_t *gt = onnx_tensor_alloc_from_file(output_path);
  printf("=====================\n");
  onnx_run(ctx);
  printf("=====================\n");

  int b = onnx_tensor_equal(gt,output);
  printf("same %d\n",b);
}

int main(int argc, char** argv){
  test_slice(
      join(test_base_dir,"test_slice/model.onnx")
      , join(test_base_dir, "test_slice/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice/test_data_set_0/output_0.pb")
      );

  // default_axes
  // printf("%s\n",join(test_base_dir,"test_slice_default_axes/model.onnx"));
  test_slice(
      join(test_base_dir,"test_slice_default_axes/model.onnx")
      , join(test_base_dir, "test_slice_default_axes/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_default_axes/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_default_axes/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_default_axes/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_default_axes/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_default_axes/test_data_set_0/output_0.pb")
      );
  //
  test_slice(
      join(test_base_dir,"test_slice_default_steps/model.onnx")
      , join(test_base_dir, "test_slice_default_steps/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_default_steps/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_default_steps/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_default_steps/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_default_steps/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_default_steps/test_data_set_0/output_0.pb")
      );

  test_slice(
        join(test_base_dir, "test_slice_end_out_of_bounds/model.onnx")
      , join(test_base_dir, "test_slice_end_out_of_bounds/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_end_out_of_bounds/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_end_out_of_bounds/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_end_out_of_bounds/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_end_out_of_bounds/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_end_out_of_bounds/test_data_set_0/output_0.pb")
      );


  test_slice(
        join(test_base_dir, "test_slice_neg/model.onnx")
      , join(test_base_dir, "test_slice_neg/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_neg/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_neg/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_neg/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_neg/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_neg/test_data_set_0/output_0.pb")
      );


  test_slice(
        join(test_base_dir, "test_slice_negative_axes/model.onnx")
      , join(test_base_dir, "test_slice_negative_axes/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_negative_axes/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_negative_axes/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_negative_axes/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_negative_axes/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_negative_axes/test_data_set_0/output_0.pb")
      );

  test_slice(
        join(test_base_dir, "test_slice_neg_steps/model.onnx")
      , join(test_base_dir, "test_slice_neg_steps/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_neg_steps/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_neg_steps/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_neg_steps/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_neg_steps/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_neg_steps/test_data_set_0/output_0.pb")
      );

  test_slice(
        join(test_base_dir, "test_slice_start_out_of_bounds/model.onnx")
      , join(test_base_dir, "test_slice_start_out_of_bounds/test_data_set_0/input_0.pb")
      , join(test_base_dir, "test_slice_start_out_of_bounds/test_data_set_0/input_1.pb")
      , join(test_base_dir, "test_slice_start_out_of_bounds/test_data_set_0/input_2.pb")
      , join(test_base_dir, "test_slice_start_out_of_bounds/test_data_set_0/input_3.pb")
      , join(test_base_dir, "test_slice_start_out_of_bounds/test_data_set_0/input_4.pb")
      , join(test_base_dir, "test_slice_start_out_of_bounds/test_data_set_0/output_0.pb")
      );
  return 0;
}
