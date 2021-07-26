import torch
from torch.onnx.utils import _model_to_graph
from torch.onnx.symbolic_helper import _set_onnx_shape_inference

model= torch.jit.load('/media/sanjiban/Applications2/GSoC21/Test/PyTorch/PyTorchModelModule.pt')
model.cpu()
model.eval()

tem=[120,1]

dummy=torch.rand(120,1) #check for multiple inputs
op=model(dummy)


_set_onnx_shape_inference(True)
graph=_model_to_graph(model,dummy,example_outputs=op)

modelData=[]
for i in graph[0].nodes():
  nodeData=[]
  nodeData.extend([x.debugName() for x in i.outputs()]) 
  nodeData.extend([x.type().scalarType() for x in i.outputs()])
  nodeData.extend([x.type().sizes() for x in i.outputs()]) 
  modelData.append(nodeData)



weightNames=[k for k in graph[1].keys()]
weights=[v.numpy() for v in graph[1].values()]

print(weightNames)
print(weights)
