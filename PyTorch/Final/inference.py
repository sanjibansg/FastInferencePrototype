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


from torch.onnx.utils import _model_to_graph

import torch
import torch.nn as nn


model=torch.jit.load('model.pt')
dummy=torch.rand(1,3, 224,224)

op=model(dummy)
#torch.onnx.export(model,dummy,"model.onnx",example_outputs=op)

model.eval()


graph= _model_to_graph(model,dummy,example_outputs=op)


for i in graph[0].inputs():
 print(i.debugName())

for i in graph[0].nodes():
 print(i.kind())




