
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
'''
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
#m = torch.jit.script(model)    


def _model_to_graph(model, args):
    if isinstance(args, torch.Tensor):
        args = (args, )
    graph = model.forward.graph
    method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
    in_vars, in_desc = torch.jit._flatten(tuple(args) + tuple(params))
    graph = _propagate_and_assign_input_shapes(  method_graph, tuple(in_vars), False, False)
    return graph

import torchvision
from torch._C import _propagate_and_assign_input_shapes
'''
'''
import torchvision
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
dummy = torch.rand(1,3,224,224)
m = torch.jit.script(model)    
'''
'''
#traced = torch.jit.trace(model, dummy)
#graph=_model_to_graph(traced,dummy)
#print(graph)

for i in graph.inputs():
 print(i.type().dim(),end=" ")
 

for i in graph.nodes():
 print(i.kind())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x=self.conv(x)
        return self.fc1(x)

n = Net()
#model=torch.jit.load('model.pt')
'''  


#print(dir(m.graph))
#for i in m.graph.nodes():
# print(dir(i))
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 120)

    def forward(self, x):
        x=self.fc1(x)
        return x
model = Net()
model.eval()  

from torch.onnx.utils import _model_to_graph
dummy=torch.rand(2,2,3,3)
op=model(dummy)
'''
#torch.onnx.export(model,dummy,"net.onnx",example_outputs=op)
#args=dummy
#args = _decide_input_format(model, dummy)
#op=model(dummy)

#graph= _model_to_graph(model,args,example_outputs=op)
#graph=torch.jit.trace(graph,example_forward_input)
#print(type(graph[0]))

#model=model.float()
#print(model)
'''
for i in model.graph.inputs():
 print(i.debugName())
'''
#graph=_model_to_graph(model,args,example_outputs=op)
#graph1=_model_to_graph(model,args,example_outputs=op)
#import torchvision
from torch.onnx.utils import _model_to_graph
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 120)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.fc1(x)
        y=self.relu(x)
        return y
model = Net()
model.eval()  
#import torchvision
#model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
m = torch.jit.script(model)    
torch.jit.save(m, "model.pt")
# Initialize model
#model = TheModelClass()
#m = torch.jit.script(model)  
'''  
graph=torch.jit.script(model)
graph.eval()
dummy=torch.rand(2,2,3,3)
op=model(dummy)
args=dummy
graph=_model_to_graph(model,args,example_outputs=op)
print(graph)
'''
'''
traced=torch.jit.trace(model,dummy)
#print(graph[0])
for op in traced.graph.outputs():
  print(op.debugName())
  print(str(op.type())+" "+str(op.type().scalarType()))



'''
