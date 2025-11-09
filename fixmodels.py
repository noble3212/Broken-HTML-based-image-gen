import onnx

# Load your original ONNX (without weights)
model = onnx.load("model.onnx")

# Load weights from the .pb (assuming you have a way to extract tensors)
# For Stable Diffusion exported with tf2onnx, usually the .pb has the same node names
# You would map them using onnx numpy arrays and assign to model.graph.initializer

# Save new ONNX with embedded weights
onnx.save(model, "model_with_weights.onnx")