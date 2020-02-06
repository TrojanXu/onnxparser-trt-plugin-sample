import onnxruntime
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
import torch
import torch.nn.functional as F

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()

    def forward(self, input, grid):
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

def export_onnx_model(onnx_model_file):
    torch_input = torch.rand(2, 1, 4, 4) # N C H W
    torch_grid = torch.rand(2, 4, 4, 2)

    model = MyModel()
    torch.onnx.export( model, (torch_input, torch_grid), onnx_model_file, verbose=True, input_names=['input', 'grid'],output_names=['output'],opset_version =10)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)

if __name__=='__main__':

    onnx_model_file = "grid_sample.onnx"
    export_onnx_model(onnx_model_file)
    
    input = np.random.rand(2, 1, 4, 4).astype('float32')
    grid = np.random.rand(2, 4, 4, 2).astype('float32')

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            inputs[0].host = input
            inputs[1].host = grid
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print(trt_outputs)
