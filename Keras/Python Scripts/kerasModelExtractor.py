from keras.models import load_model
#Loading model and weights
filepath="model.h5"
kerasModel=load_model(filepath)
kerasModel.load_weights(filepath)
print(kerasModel.summary())
globals().update(locals())


modelData=[]
for idx in range(len(kerasModel.layers)):
    layerData={}
    globals().update(locals())
    layerData.update({(k,v) for (k,v) in {key:getattr(value,'__name__',None) for (key,value)  in {i:getattr(kerasModel.get_layer(index=idx),i,None) for i in ['__class__','activation']}.items()}.items()})
    layerData.update({(k,v) for (k,v) in {i:getattr(kerasModel.get_layer(index=idx),i,None) for i in ['name','dtype','dims']}.items()})
    layerData.update({(k,v) for (k,v) in {key:getattr(value,'name',None) for (key,value)  in {i:getattr(kerasModel.get_layer(index=idx),i,None) for i in ['input','output','kernel','bias']}.items()}.items()})
    modelData.append(layerData)

#Extracting model weights (initialized tensors)
weight=[]
for idx in range(len(kerasModel.get_weights())):
    weightProp={}
    weightProp['name']=kerasModel.weights[idx].name
    weightProp['dtype']=(kerasModel.get_weights())[idx].dtype.name
    weightProp['value']=(kerasModel.get_weights())[idx]
    weight.append(weightProp)
    
print(weight)

inputNames=kerasModel.input_names
inputShapes=kerasModel.input_shape
inputTypes=[]
for idx in range(len(kerasModel.inputs)):
    inputTypes.append(kerasModel.inputs[idx].dtype.__str__()[9:-2])
    
    

print(inputTypes)


#Extracting model information
'''
modelData=[]
for idx in range(len(kerasModel.layers)):
    layerData={}
    layerData.update({(k,v) for (k,v) in {key:getattr(value,'__name__',None) for (key,value)  in {i:getattr(kerasModel.get_layer(index=idx),i,None) for i in ['__class__','activation']}.items()}.items()})
    layerData.update({(k,v) for (k,v) in {i:getattr(kerasModel.get_layer(index=idx),i,None) for i in ['name','dtype','dims']}.items()})
    layerData.update({(k,v) for (k,v) in {key:getattr(value,'name',None) for (key,value)  in {i:getattr(kerasModel.get_layer(index=idx),i,None) for i in ['input','output','kernel','bias']}.items()}.items()})
    modelData.append(layerData)
#Extracting model weights (initialized tensors)
weight=[]
for idx in range(len(kerasModel.get_weights())):
    weightProp={}
    weightProp['name']=kerasModel.weights[idx].name
    weightProp['dtype']=(kerasModel.get_weights())[idx].dtype.name
    weightProp['value']=(kerasModel.get_weights())[idx]
    weight.append(weightProp)
#Extracing input tensors info
inputNames=kerasModel.input_names
inputShapes=kerasModel.input_shape
inputTypes=[]
for idx in range(len(kerasModel.inputs)):
    inputTypes.append(kerasModel.inputs[idx].dtype.__str__()[9:-2])
    
    
#print(modelData)
#print(weight)
print(inputTypes)
'''
