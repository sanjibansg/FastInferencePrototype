from keras.models import load_model

def loadModel(path):
	model=load_model(path)
	model.load_weights(path)
	modelProp=[]
	for idx in range(len(model.layers)):
		layerProp=[]
		layerProp.extend([x for x in [getattr(obj,'__name__',None) for obj in [getattr(model.get_layer(index=idx),i,None) for i in ['__class__','__activation__']]]])
		layerProp.extend([x for x in [getattr(obj,'name',None) for obj in [getattr(model.get_layer(index=idx),i,None) for i in ['input','output']]]])
		layerProp.extend([x for x in [getattr(model.get_layer(index=idx),i,None) for i in ['input_shape','output_shape'] ]])
		modelProp.append(layerProp)
	return modelProp,model.get_weights()
