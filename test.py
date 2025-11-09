import tkinter as tk
from tkinter import filedialog, messagebox
import onnx
import tensorflow as tf

def load_onnx():
    file_path = filedialog.askopenfilename(title="Select ONNX model", filetypes=[("ONNX files", "*.onnx")])
    if file_path:
        try:
            model = onnx.load(file_path)
            messagebox.showinfo("Success", f"ONNX model loaded with {len(model.graph.node)} nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ONNX model:\n{e}")

def load_pb():
    file_path = filedialog.askopenfilename(title="Select TensorFlow PB file", filetypes=[("PB files", "*.pb")])
    if file_path:
        try:
            # Try to load as frozen graph
            with tf.io.gfile.GFile(file_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            messagebox.showinfo("Success", f"PB file loaded with {len(graph_def.node)} nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PB file:\n{e}")

root = tk.Tk()
root.title("Model Loader")

btn_onnx = tk.Button(root, text="Load ONNX Model", command=load_onnx)
btn_onnx.pack(pady=10)

btn_pb = tk.Button(root, text="Load PB File", command=load_pb)
btn_pb.pack(pady=10)

root.mainloop()
