import torch
from torch.onnx.utils import _model_to_graph
'''
model= torch.jit.load('model.pt')
model.eval()
dummy=torch.rand(1,3,224,224) #check for multiple inputs
op=model(dummy)
'''
#For building the exampleple inputs, iterate over the vector of input shapes, create a tuple from there, and then create torch.rand () using them, and populate a list args using them, which will be passed to be used by the _model_to_graph


model= torch.jit.load('model.pt')
model.eval()
model.float()
tem=[120,1]
dummy=torch.rand(*tem) #check for multiple inputs
tex=[dummy]
op=model(*tex)

args=dummy
graph=_model_to_graph(model,tex,example_outputs=op)




#Extracting model operators information
modelData=[]
for i in graph[0].nodes():
 nodeData=[]
 nodeData.append(i.kind())
 nodeAttributeNames=[x for x in i.attributeNames()]
 nodeAttributes={j:i[j] for j in nodeAttributeNames}
 nodeData.append(nodeAttributes)
 nodeInputs=[x for x in i.inputs()]
 nodeInputNames=[x.debugName() for x in nodeInputs]
 nodeData.append(nodeInputNames)
 nodeOutputs=[x for x in i.outputs()]
 nodeOutputNames=[x.debugName() for x in nodeOutputs]
 nodeData.append(nodeOutputNames)
 #nodeDType=[x.type().scalarType() for x in nodeOutputs]
 #nodeData.append(nodeDType)
 modelData.append(nodeData)
 
weightNames=[k for k in graph[1].keys()] 
weights=[v.numpy() for v in graph[1].values()]

inputs=[x for x in model.graph.inputs()]
inputs=inputs[1:]
inputNames=[x.debugName() for x in inputs]

outputs=[x for x in graph[0].outputs()]
outputNames=[x.debugName() for x in outputs]
print(outputNames)
''' 
#Extracting model weights
weights={k:v.numpy() for k,v in graph[1].items()}
#print(weights) 
## find dtype pf weights by using PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)¶
## and char PyArray_Descr.kind¶

inputs=[x for x in graph[0].inputs()]
#inputs=inputs[1:]
inputNames=[x.debugName() for x in inputs]
inputTypes=[x.type().scalarType() for x in inputs]
#print(inputNames)

outputs=[x for x in graph[0].outputs()]
outputNames=[x.debugName() for x in outputs]
outputTypes=[x.type().scalarType() for x in outputs]
print(outputNames)
print(outputTypes)
print(inputNames)
print(inputTypes)
'''
#print(modelData)
