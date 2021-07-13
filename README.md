# FastInferencePrototype
Fast Inference of Machine Learning Models following the ONNX standards in ROOT/TMVA SOFIE (“System for Optimized Fast Inference code Emit”)

### Keras
 Maps keras layers to equivalent graph operations and returns graph nodes connected with node inputs and corresponding nodes names and the model weights in list of numpy array format.

### PyTorch
Converts a model scripted using Torchscript to its equivalent ONNX graph using dummy inputs, and utility functions. Using the ONNX graph, model data and weights are extracted.

### tmva
Directory for Toolkit for Multivariate Analysis (TMVA) of the Root-Project.
PyConverters: Functionality for converting PyTorch and keras models to .root files.

### restructure
Restructured SOFIE to avoid dependency conflicts with required versions of Protobuf

### serialisation
Serialising RModel and ROperators for saving them into ROOT format.

### Collections
Collections of relevant documents, manuals, user-guides for the project.
