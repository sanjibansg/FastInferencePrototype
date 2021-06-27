import torch



#model = torch.load('model.pt')
from torch._C import _propagate_and_assign_input_shapes

#model=torch.load('model.pt')
model=torch.jit.load('model.pt')
dummy=torch.rand((1,3,224,224))
#model=torch.jit.trace(model,dummy)
#mod=torch.jit.trace(model,dummy)
#print(dir(model))
#print(model.named_children)
'''
for idx, m in enumerate(model.modules()):
        print(idx, '->', m)

'''

'''
for i in model.children():
 if isinstance(i,torch.nn.Linear):
  print(i)
 for j in i.graph.inputs():
  print(j.debugName())
  
@torch.no_grad()
def init_weights(m):    
     if isinstance(m,torch.nn.ReLU):
         print(m)
         
model.apply(init_weights)  
#model=torch.jit.trace(model.code, dummy)
'''
def _model_to_graph(model, args):
    if isinstance(args, torch.Tensor):
        args = (args, )
    graph = model.forward.graph
    method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
    args_params = tuple(args) + tuple(params)
    in_vars, in_desc = torch.jit._flatten(tuple(args) + tuple(params))
    graph = _propagate_and_assign_input_shapes(method_graph, tuple(in_vars), False, False)
    return graph
model.eval()
frozen=torch.jit.freeze(model)    
graph=_model_to_graph(model,dummy)
'''
model.eval()

scripted=torch.jit.script(model.eval())
frozen_module = torch.jit.freeze(traced)

frozen_module=scripted
print(type(frozen_module))

for i in frozen_module.named_buffers():
 print(i)


#print(model.graph.inputs())
for i in frozen.graph.inputs():
 print(i.debugName())

#print(dir(frozen_module.graph))

#print(dir(frozen_module.graph.op))
for i in frozen.graph.nodes():
 print(i.kind())
'''
