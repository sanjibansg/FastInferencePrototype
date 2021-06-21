import torch



#model = torch.load('model.pt')
from torch._C import _propagate_and_assign_input_shapes
model=torch.jit.load('model.pt')

dummy=torch.rand((1,3,224,224))
traced=torch.jit.trace(model,dummy)
'''
def _model_to_graph(model, args):
    if isinstance(args, torch.Tensor):
        args = (args, )
    graph = model.forward.graph
    method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
    in_vars, in_desc = torch.jit._flatten(tuple(args) + tuple(params))
    graph = _propagate_and_assign_input_shapes(  method_graph, tuple(in_vars), False, False)
    return graph
    
graph=_model_to_graph(model,dummy)

model.eval()

scripted=torch.jit.script(model.eval())
'''
frozen_module = torch.jit.freeze(traced)
'''
frozen_module=scripted
print(type(frozen_module))

for i in frozen_module.named_buffers():
 print(i)
'''

#print(model.graph.inputs())
for i in frozen_module.graph.inputs():
 print(i.debugName())

#print(dir(frozen_module.graph))

#print(dir(frozen_module.graph.op))
for i in frozen_module.graph.nodes():
 print(i.kind())
 
