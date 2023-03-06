#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "src/onnx.h"
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct CenterPrior
{
    int x;
    int y;
    int stride;
} CenterPrior;

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

static int generate_grid_center_priors(const int input_height, const int input_width, int* strides, int n_strides, CenterPrior* center_priors, int max_centers){
  assert(center_priors != NULL);
  int index = 0;
  for(int i = 0; i < n_strides; ++i){
    int stride = strides[i];
    int feat_w = ceil((float)input_width / stride);
    int feat_h = ceil((float)input_height / stride);
    for (int y = 0; y < feat_h; y++){
      for (int x = 0; x < feat_w; x++){
        center_priors[index].x = x;
        center_priors[index].y = y;
        center_priors[index].stride = stride;
        ++index;
      }
    }
  }
  assert(index < max_centers);
  return index;
}

int read_jpeg_file(const char* input_filename, char **output_buffer){
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE *input_file;
  FILE *output_file;
  JSAMPARRAY buffer;
  int row_width;
  unsigned char *rowdata = NULL;
  cinfo.err = jpeg_std_error(&jerr);
  if((input_file = fopen(input_filename, "rb"))==NULL){
    fprintf(stderr, "can not open %s\n",input_filename);
    return -1;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, input_file);

  (void)jpeg_read_header(&cinfo,TRUE);

  (void)jpeg_start_decompress(&cinfo);
  row_width = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo,JPOOL_IMAGE,row_width,1);
  *output_buffer = (unsigned char*)malloc(row_width * cinfo.output_height);
  memset(*output_buffer,0,row_width* cinfo.output_height);
  rowdata = *output_buffer;
  while(cinfo.output_scanline < cinfo.output_height){
    (void) jpeg_read_scanlines(&cinfo,buffer,1);
    memcpy(rowdata,*buffer,row_width);
    rowdata += row_width;
  }
  (void) jpeg_destroy_decompress(&cinfo);
  fclose(input_file);
  return 0;
}

int read_jpeg_file_stb(const char* input_filename, char **output_buffer){
  int iw,ih,n;
  unsigned char*idata = stbi_load(input_filename, &iw, &ih, &n, 0);
  printf("w=%d, h=%d, c=%d\n",iw,ih,n);
  unsigned char* odata = (unsigned char*)malloc(416*416*n);
  stbir_resize(idata, iw, ih, 0, odata, 416, 416, 0, STBIR_TYPE_UINT8, n, STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_BOX, STBIR_FILTER_BOX, STBIR_COLORSPACE_SRGB, NULL);
  const char* output_path = "./output.jpg";
  stbi_write_jpg(output_path, 416, 416, n, odata, 0);
  stbi_image_free(idata);
  // stbi_image_free(odata);
  *output_buffer = odata;
  return 0;
}

static void onnx_tensor_apply_image(struct onnx_tensor_t* y, const char * file){
  char *buffer = NULL;
  // read_jpeg_file(file,&buffer);
  read_jpeg_file_stb("/home/lzy/Code/C/onnx-learn/cat.jpeg", &buffer);
  printf("buffer size=%lu\n",strlen((const char *)buffer));
  for(int i = 0; i < y->ndim; ++i){
    printf("%d ",y->dims[i]);
  }
  printf("\n");
  printf("%ld\n",y->ndata);
  float* py = y->datas;
  int l = strlen((const char*)buffer);
  for(int i = 0; i < l; ++i){
    switch (i%3) {
      case 0:
        py[i] = ((float)buffer[i] - 103.53f)*0.017429f;
      case 1:
        py[i] = ((float)buffer[i] - 116.28f)*0.017507f;
      case 2:
        py[i] = ((float)buffer[i] - 123.675f)*0.017125f;
    }
  }
  free(buffer);
}



int main(int argc, char** argv){
  printf("123\n");
  struct onnx_context_t * ctx;
  struct onnx_tensor_t* input;
  struct onnx_tensor_t* output;
  ctx = onnx_context_alloc_from_file("/home/lzy/Code/C/onnx-learn/nanodet.onnx",NULL,0);
  // ctx = onnx_context_alloc_from_file("/home/lzy/Code/C/onnx-learn/resnet18-v1-7.onnx",NULL,0);
  printf("aclloc ctx %d\n",ctx==NULL);
  if(!ctx) return -1;
  input = onnx_tensor_search(ctx,"data");
  output = onnx_tensor_search(ctx,"output");
  printf("input name %s\n",input->name);
  onnx_tensor_apply_image(input, "/home/lzy/DATA/datasets/PRW/PRW-v16.04.20/frames/c1s1_000151.jpg");
  // onnx_tensor_apply(ctx, void *buf, size_t len)
  onnx_run(ctx);
  printf("output ndata %ld\n",output->ndata);
  int maxidx = -1;
  float maxf = -1;
  printf("%d\n",output->ndim);
  return 0;
}
