import torch
from torch import nn
import torch
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features3 = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128*128*3 + 32*32*3 + 5, 4)
        
    def forward(self, x1, x2, x3):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x3 = self.features3(x3)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x

model = MyModel()
batch_size = 1
x1 = torch.randn(batch_size, 1, 256, 256)
x2 = torch.randn(batch_size, 1, 64, 64)
x3 = torch.randn(batch_size, 10)

output = model(x1, x2, x3)


'''
summary = OrderedDict()
hooks = []

def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input']= locals()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            #summary[m_key]['weight']=getattr(module,'weight',None)
            #summary[m_key]['inputs']=dir(module)
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))
            

#device=torch.device('cuda:0')
input_size=[(1,256,256),(1,64,64),(10,)]
dtypes = [torch.FloatTensor]*len(input_size)
x = [torch.rand(2, *in_size).type(dtype)
         for in_size, dtype in zip(input_size, dtypes)]
        
model.apply(register_hook)  
model(*x)        
print(summary)  
for layer in summary:
 print(layer)
'''
from torchinfo import summary
#summary(model,input_data=[x1,x2,x3])

scripted=torch.jit.script(MyModel().eval())
frozen_module = torch.jit.freeze(scripted)
for i in frozen_module.named_buffers():
 print(i)
print(frozen_module.graph.inputs())
for i in frozen_module.graph.inputs():
 print(i.debugName())
for i in frozen_module.graph.nodes():
 for j in i.blocks():
  for k in j.nodes():
   print(dir(k))
print(frozen_module.graph)   

