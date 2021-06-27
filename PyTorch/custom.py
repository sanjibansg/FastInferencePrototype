import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch._C import _propagate_and_assign_input_shapes
import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
import collections
import contextlib
import numbers
import warnings
from torch._six import string_classes
from torch.jit import _unique_state_dict
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes, TrainingMode
from torch._C import ListType, OptionalType, _propagate_and_assign_input_shapes, _check_onnx_proto
from typing import Union, Tuple, List


def _to_graph(model, args):
    if isinstance(args, torch.Tensor):
        args = (args, )
    graph = model.forward.graph
    method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
    in_vars, in_desc = torch.jit._flatten(tuple(args) + tuple(params))
    graph = _propagate_and_assign_input_shapes(  method_graph, in_vars,in_desc, False, False)
    return graph
'''
traced_model_savepath = 'traced.pt'

'''
from torch.onnx.utils import _model_to_graph,select_model_mode_for_export,_decide_keep_init_as_input,_decide_add_node_names,_decide_constant_folding,_decide_external_data_format
from torch.onnx.utils import _decide_input_format,_validate_dynamic_axes

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 120)

    def forward(self, x):
        return self.fc1(x)

n = Net()
model=torch.jit.load('model.pt')
dummy=torch.rand(1,3,224,224)
example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 1, 3, 3)
torch.onnx.export(n,example_forward_input,"net.onnx")

# Trace a specific method and construct `ScriptModule` with
# a single `forward` method


#dummy_input = torch.rand(1,3,224,224)
#model.eval()
op=n(example_forward_input)
#traced_model = torch.jit.trace(n, example_forward_input)
#traced_model.save(traced_model_savepath)
# load the saved traced model from disk
#loaded_traced_model = torch.jit.load(traced_model_savepath)



model = torch.jit.load('model.pt')

dummy=torch.rand(1,3,224,224)
op=model(dummy)

#torch.onnx.export(model,dummy,"net.onnx",example_outputs=op)
args=dummy
#args = _decide_input_format(model, dummy)
op=model(dummy)

graph= _model_to_graph(model,args,example_outputs=op)
#graph=torch.jit.trace(graph,example_forward_input)
#print(type(graph[0]))



for i in graph[0].inputs():
 print(i.debugName())

for i in graph[0].nodes():
 print(i.kind())




'''
__IN_ONNX_EXPORT = False

def _export(model, args,f="mod.onnx",export_params=True, verbose=False, training=None,
            input_names=None, output_names=None, operator_export_type=None,
            export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None,
            opset_version=None, _retain_param_name=False, do_constant_folding=True,
            strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None,
            fixed_batch_size=False, custom_opsets=None, add_node_names=True,
            enable_onnx_checker=True, use_external_data_format=False,
            onnx_shape_inference=True):

    if isinstance(model, torch.nn.DataParallel):
        raise ValueError("torch.nn.DataParallel is not supported by ONNX "
                         "exporter, please use 'attribute' module to "
                         "unwrap model from torch.nn.DataParallel. Try "
                         "torch.onnx.export(model.module, ...)")
    global __IN_ONNX_EXPORT
    assert __IN_ONNX_EXPORT is False
    __IN_ONNX_EXPORT = True
    try:
        from torch.onnx.symbolic_helper import _set_onnx_shape_inference
        _set_onnx_shape_inference(onnx_shape_inference)

        from torch.onnx.symbolic_helper import _default_onnx_opset_version, _set_opset_version
        from torch.onnx.symbolic_helper import _set_operator_export_type
        if opset_version is None:
            opset_version = _default_onnx_opset_version
        if not operator_export_type:
            if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
                operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
            else:
                operator_export_type = OperatorExportTypes.ONNX

        # By default, training=None, (which defaults to TrainingMode.EVAL),
        # which is good because running a model in training mode could result in
        # internal buffers getting updated, dropout getting applied, etc.
        # If you really know what you're doing, you can turn
        # training=TrainingMode.TRAINING or training=TrainingMode.PRESERVE,
        # (to preserve whatever the original training mode was.)
        _set_opset_version(opset_version)
        _set_operator_export_type(operator_export_type)
        with select_model_mode_for_export(model, training):
            val_keep_init_as_ip = _decide_keep_init_as_input(keep_initializers_as_inputs,
                                                             operator_export_type,
                                                             opset_version)
            val_add_node_names = _decide_add_node_names(add_node_names, operator_export_type)
            val_do_constant_folding = _decide_constant_folding(do_constant_folding, operator_export_type, training)
            val_use_external_data_format, model_file_location = _decide_external_data_format(use_external_data_format,
                                                                                             operator_export_type,
                                                                                             f)
            args = _decide_input_format(model, args)
            if dynamic_axes is None:
                dynamic_axes = {}
            _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

            graph, params_dict, torch_out = \
                _model_to_graph(model, args, verbose, input_names,
                                output_names, operator_export_type,
                                example_outputs, _retain_param_name,
                                val_do_constant_folding,
                                fixed_batch_size=fixed_batch_size,
                                training=training,
                                dynamic_axes=dynamic_axes)

            # TODO: Don't allocate a in-memory string for the protobuf
            defer_weight_export = export_type is not ExportTypes.PROTOBUF_FILE
            if custom_opsets is None:
                custom_opsets = {}

            if export_params:
                proto, export_map = graph._export_onnx(
                    params_dict, opset_version, dynamic_axes, defer_weight_export,
                    operator_export_type, strip_doc_string, val_keep_init_as_ip, custom_opsets,
                    val_add_node_names, val_use_external_data_format, model_file_location)
            else:
                proto, export_map = graph._export_onnx(
                    {}, opset_version, dynamic_axes, False, operator_export_type,
                    strip_doc_string, val_keep_init_as_ip, custom_opsets, val_add_node_names,
                    val_use_external_data_format, model_file_location)

        return proto,export_map
    finally:
        assert __IN_ONNX_EXPORT
        __IN_ONNX_EXPORT = False


graph,hmap=_export(model,example_forward_input,example_outputs=op)
'''
