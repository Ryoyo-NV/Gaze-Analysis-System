#include <Python.h>
#include <pygobject.h>
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gmodule.h>
#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"
#include "nvds_version.h"
#include "nvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsinfer.h"
#include "cuda_runtime_api.h"
#include "ds_facialmark_meta.h"
#include "ds_gaze_meta.h"
#include "cv/core/Tensor.h"
#include "nvbufsurface.h"
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using std::string;

std::unique_ptr<cvcore::faciallandmarks::FacialLandmarksPostProcessor> facemarkpost;

PyObject* init_fpe_postprocess(PyObject* self, PyObject* args, PyObject* kw)
{
    static const char* argkws[] = {"num", "max_bsize", "in_width", "in_height", NULL};
    size_t num=80, max_bsize=32, in_width=80, in_height=80;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|iiii", const_cast<char**>(argkws),
            &num, &max_bsize, &in_width, &in_height))
        return NULL;
    
    size_t numFaciallandmarks = num;
    cvcore::ModelInputParams ModelInputParams = {max_bsize, in_width, in_height, cvcore::Y_F32};

    std::unique_ptr< cvcore::faciallandmarks::FacialLandmarksPostProcessor > faciallandmarkpostinit(
    new cvcore::faciallandmarks::FacialLandmarksPostProcessor (
    ModelInputParams,numFaciallandmarks));
    facemarkpost = std::move(faciallandmarkpostinit);

    return Py_BuildValue("i", GST_PAD_PROBE_OK);
}

/*Generate bodypose2d display meta right after inference */
static GstPadProbeReturn
tile_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  int part_index = 0;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);    
    NvDsDisplayMeta *disp_meta = NULL;
 
    if (!frame_meta)
      continue;

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      if (!obj_meta)
        continue;

      bool facebboxdraw = false;
        
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list;
          l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if(user_meta->base_meta.meta_type ==
            (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK) {
          NvDsFacePointsMetaData *facepoints_meta =
              (NvDsFacePointsMetaData *)user_meta->user_meta_data;
          /*Get the face marks and mark with dots*/
          if (!facepoints_meta)
            continue;
          for (part_index = 0;part_index < facepoints_meta->facemark_num;
              part_index++) {
            if (!disp_meta) {
              disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
              disp_meta->num_circles = 0;
              disp_meta->num_rects = 0;
              
            } else {
              if (disp_meta->num_circles==MAX_ELEMENTS_IN_DISPLAY_META) {
                
                nvds_add_display_meta_to_frame (frame_meta, disp_meta);
                disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                disp_meta->num_circles = 0;
              }
            }
            if(!facebboxdraw) {
              disp_meta->rect_params[disp_meta->num_rects].left =
                facepoints_meta->right_eye_rect.left +
                obj_meta->rect_params.left;
              disp_meta->rect_params[disp_meta->num_rects].top =
                facepoints_meta->right_eye_rect.top +
                obj_meta->rect_params.top;
              disp_meta->rect_params[disp_meta->num_rects].width =
                facepoints_meta->right_eye_rect.right -
                facepoints_meta->right_eye_rect.left;
              disp_meta->rect_params[disp_meta->num_rects].height =
                facepoints_meta->right_eye_rect.bottom -
                facepoints_meta->right_eye_rect.top;
              disp_meta->rect_params[disp_meta->num_rects].border_width = 2;
              disp_meta->rect_params[disp_meta->num_rects].border_color.red
                = 1.0;
              disp_meta->rect_params[disp_meta->num_rects].border_color.green
                = 1.0;
              disp_meta->rect_params[disp_meta->num_rects].border_color.blue
                = 0.0;
              disp_meta->rect_params[disp_meta->num_rects].border_color.alpha
                = 0.5;
              disp_meta->rect_params[disp_meta->num_rects+1].left =
                facepoints_meta->left_eye_rect.left + obj_meta->rect_params.left;
              disp_meta->rect_params[disp_meta->num_rects+1].top =
                facepoints_meta->left_eye_rect.top + obj_meta->rect_params.top;
              disp_meta->rect_params[disp_meta->num_rects+1].width =
                facepoints_meta->left_eye_rect.right -
                facepoints_meta->left_eye_rect.left;
              disp_meta->rect_params[disp_meta->num_rects+1].height =
                facepoints_meta->left_eye_rect.bottom -
                facepoints_meta->left_eye_rect.top;
              disp_meta->rect_params[disp_meta->num_rects+1].border_width = 2;
              disp_meta->rect_params[disp_meta->num_rects+1].border_color.red
                = 1.0;
              disp_meta->rect_params[disp_meta->num_rects+1].border_color
               .green = 1.0;
              disp_meta->rect_params[disp_meta->num_rects+1].border_color
               .blue = 0.0;
              disp_meta->rect_params[disp_meta->num_rects+1].border_color
               .alpha = 0.5;
              disp_meta->num_rects+=2;
              facebboxdraw = true;
            }

            disp_meta->circle_params[disp_meta->num_circles].xc =
                facepoints_meta->mark[part_index].x + obj_meta->rect_params.left;
            disp_meta->circle_params[disp_meta->num_circles].yc =
                facepoints_meta->mark[part_index].y + obj_meta->rect_params.top;
            disp_meta->circle_params[disp_meta->num_circles].radius = 1;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.red = 0.0;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.green = 1.0;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.blue = 0.0;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.alpha = 0.5;
            disp_meta->num_circles++;
          }
        }
      }
    }
    if (disp_meta && disp_meta->num_circles)
       nvds_add_display_meta_to_frame (frame_meta, disp_meta);
  }
  return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. The face bbox will be scale to square for
 * facial marks.
 */
static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));
  NvBufSurface *in_surf;
  GstMapInfo in_map_info;
  int frame_width, frame_height;

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (GST_BUFFER (info->data), &in_map_info, GST_MAP_READ)) {
    g_printerr ("Failed to map GstBuffer\n");
    return GST_PAD_PROBE_PASS;
  }
  in_surf = (NvBufSurface *) in_map_info.data;
  gst_buffer_unmap(GST_BUFFER (info->data), &in_map_info);

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    frame_width = in_surf->surfaceList[frame_meta->batch_id].width;
    frame_height = in_surf->surfaceList[frame_meta->batch_id].height;
    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {

      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
      
      if (!obj_meta) {
        g_print("No obj meta\n");
        break;
      }
      if(obj_meta->rect_params.left<0)
          obj_meta->rect_params.left=0;
      if(obj_meta->rect_params.top<0)
          obj_meta->rect_params.top=0;
          
      float square_size = MAX(obj_meta->rect_params.width,
          obj_meta->rect_params.height);
      float center_x = obj_meta->rect_params.width/2.0 +
          obj_meta->rect_params.left;
      float center_y = obj_meta->rect_params.height/2.0 +
          obj_meta->rect_params.top;

      /*Check the border*/
      if(center_x < (square_size/2.0) || center_y < square_size/2.0 ||
          center_x + square_size/2.0 > frame_width ||
          center_y - square_size/2.0 > frame_height) {
              //g_print("Keep the original bbox\n");
	      continue;
      } else {
          obj_meta->rect_params.left = center_x - square_size/2.0;
          obj_meta->rect_params.top = center_y - square_size/2.0;
          obj_meta->rect_params.width = square_size;
          obj_meta->rect_params.height = square_size;
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the src pad
 * of the SGIE's next queue element. The facial marks output will be processed
 * and the facial marks metadata will be generated.
 */
static GstPadProbeReturn
sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {

      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
      
      if (!obj_meta)
        continue;

      /* Iterate user metadata in object to search SGIE's tensor data */
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;

        NvDsInferTensorMeta *meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;
        float * heatmap_data = NULL;
        float * confidence = NULL;
        //int heatmap_c = 0;

        for (unsigned int i = 0; i < meta->num_output_layers; i++) {
          NvDsInferLayerInfo *info = &meta->output_layers_info[i];
          info->buffer = meta->out_buf_ptrs_host[i];

          std::vector < NvDsInferLayerInfo >
            outputLayersInfo (meta->output_layers_info,
            meta->output_layers_info + meta->num_output_layers);
          //Prepare CVCORE input layers
          if (strcmp(outputLayersInfo[i].layerName,
              "softargmax") == 0) {
            //This layer output landmarks coordinates
            heatmap_data = (float *)meta->out_buf_ptrs_host[i];
          } else if (strcmp(outputLayersInfo[i].layerName,
              "softargmax:1") == 0) {
            confidence = (float *)meta->out_buf_ptrs_host[i];
          }
        }

        cvcore::Tensor<cvcore::CL, cvcore::CX, cvcore::F32> tempheatmap(
            cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS, 1,
            (float *)heatmap_data, true);
        cvcore::Array<cvcore::BBox> faceBBox(1);
        faceBBox.setSize(1);
        faceBBox[0] = {0, 0, (int)obj_meta->rect_params.width,
            (int)obj_meta->rect_params.height};
        //Prepare output array
    try{
        cvcore::Array<cvcore::ArrayN<cvcore::Vector2f,
            cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS>>
            output(1, true);
        output.setSize(1);
      
        facemarkpost->execute(output, tempheatmap, faceBBox, NULL);
      
        /*add user meta for facemark*/
        if (!nvds_add_facemark_meta (batch_meta, obj_meta, output[0],
            confidence)) {
          g_printerr ("Failed to get bbox from model output\n");
        }
    } catch (std::exception e) {
        std::cerr << e.what() << std::endl;
    }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* Python interfaces for probes
 * It's cannot be implemented in python codes because cvcore does not have
 * python interfaces and we cannot see the codes.
 */
static PyObject* cpp_facenet_pad_buffer_probe(PyObject* self, PyObject* args)
{
    PyObject* pypad = NULL;
    PyObject* pyinfo = NULL;
    PyObject* pyudata = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &pypad, &pyinfo, &pyudata))
        return NULL;

    GstPad* pad = (GstPad*) pygobject_get(pypad);
    GstPadProbeInfo* info = (GstPadProbeInfo*) pygobject_get(pyinfo);
    gpointer udata = (gpointer) pygobject_get(pyudata);

    pgie_pad_buffer_probe(pad, info, udata);

    return Py_BuildValue("i", GST_PAD_PROBE_OK);
}

static PyObject* cpp_fpenet_pad_buffer_probe(PyObject* self, PyObject* args)
{
    PyObject* pypad = NULL;
    PyObject* pyinfo = NULL;
    PyObject* pyudata = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &pypad, &pyinfo, &pyudata))
        return NULL;

    GstPad* pad = (GstPad*) pygobject_get(pypad);
    GstPadProbeInfo* info = (GstPadProbeInfo*) pygobject_get(pyinfo);
    gpointer udata = (gpointer) pygobject_get(pyudata);

    sgie_pad_buffer_probe(pad, info, udata);

    return Py_BuildValue("i", GST_PAD_PROBE_OK);
}

static PyObject* cpp_nvtile_pad_buffer_probe(PyObject* self, PyObject* args)
{
    PyObject* pypad = NULL;
    PyObject* pyinfo = NULL;
    PyObject* pyudata = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &pypad, &pyinfo, &pyudata))
        return NULL;

    GstPad* pad = (GstPad*) pygobject_get(pypad);
    GstPadProbeInfo* info = (GstPadProbeInfo*) pygobject_get(pyinfo);
    gpointer udata = (gpointer) pygobject_get(pyudata);

    tile_sink_pad_buffer_probe(pad, info, udata);

    return Py_BuildValue("i", GST_PAD_PROBE_OK);
}

static PyObject* get_gaze_from_usermeta(PyObject* self, PyObject* args)
{
    PyObject* pygazemeta = NULL;
    if (!PyArg_ParseTuple(args, "O", &pygazemeta))
        return NULL;
    
    const char* name = PyCapsule_GetName(pygazemeta);
    NvDsGazeMetaData* gazemeta = (NvDsGazeMetaData*) PyCapsule_GetPointer(pygazemeta, name);

    PyObject* pylist = PyList_New(cvcore::gazenet::GazeNet::OUTPUT_SIZE);
    for (int i = 0; i < cvcore::gazenet::GazeNet::OUTPUT_SIZE; i++)
    {
        PyObject* pyval = Py_BuildValue("f", gazemeta->gaze_params[i]);
        PyList_SetItem(pylist, i, pyval);
    }

    return pylist;
}

static PyMethodDef CppProbes[] = {
    {"facenet_pad_buffer_probe", cpp_facenet_pad_buffer_probe, METH_VARARGS, "cpp callback probe for facenet"},
    {"fpenet_pad_buffer_probe", cpp_fpenet_pad_buffer_probe, METH_VARARGS, "cpp callback probe for fpenet"},
    {"nvtile_pad_buffer_probe", cpp_nvtile_pad_buffer_probe, METH_VARARGS, "cpp callback probe for nvtile"},
    {"init_fpe_postprocess", (PyCFunction)init_fpe_postprocess, METH_VARARGS | METH_KEYWORDS, "initialize faciallandmark postprocess"},
    {"get_gaze_from_usermeta", get_gaze_from_usermeta, METH_VARARGS, "get gaze data from cpp usermeta"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef CppProbesMod = {
    PyModuleDef_HEAD_INIT,
    "dscprobes",
    "cpp functions for gst pad buffer probe",
    -1,
    CppProbes
};

PyMODINIT_FUNC PyInit_dscprobes(void)
{
    return PyModule_Create(&CppProbesMod);
}
