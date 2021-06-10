from keras.models import load_model

operator_map=[
    {'dense':['gemm','activation']},
]
activation=[
    'relu',
]
def loadModel(path):
    model=load_model(path)
    model.load_weights(path)
    graph=[]
    for layer in model.layers:
        node={}
        #insert code for expanding stacked layers
        if layer.__class__.__name__ == 'Dense':
            node['input']= layer.input.name
            if layer.activation.__name__=='linear':
               node['name']=layer.output.name
               node['type']='BiasAdd'
               graph.append(node)
            else:
                node['name']=layer.name+'BiasAdd'
                node['type']='BiasAdd'
                graph.append(node)
                node_next={}
                node_next['input']=node['name']
                node_next['name']=layer.output.name
                node_next['type']=layer.activation.__name__.capitalize()
                graph.append(node_next)

        elif layer.__class__.__name__== 'ReLU':
            node['input']=layer.input.name
            node['name']=layer.output.name
            node['type']='relu'
            graph.append(node)

        elif layer.__class__.__name__=='InputLayer':
            node['name']=layer.output.name
            node['type']='input'
            node['shape']=layer._batch_input_shape[1:]
            graph.append(node)
    print(graph)


