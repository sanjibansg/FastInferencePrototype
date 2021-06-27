import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch._C import _propagate_and_assign_input_shapes

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
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
'''
from torch.onnx.utils import _model_to_graph

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

n = Net()
model=torch.jit.load('model.pt')
dummy=torch.rand(1,3,224,224)
example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 1, 3, 3)

# Trace a specific method and construct `ScriptModule` with
# a single `forward` method


#dummy_input = torch.rand(1,3,224,224)
#model.eval()
op=n(example_forward_input)
model.eval()
#traced_model = torch.jit.trace(n, example_forward_input)
#traced_model.save(traced_model_savepath)
# load the saved traced model from disk
#loaded_traced_model = torch.jit.load(traced_model_savepath)



graph= _model_to_graph(model,example_forward_input,example_outputs=op)


print(graph)

for i in graph[0].inputs():
 print(i.debugName())
 
for i in graph[0].nodes():
 print(i.kind()) 

