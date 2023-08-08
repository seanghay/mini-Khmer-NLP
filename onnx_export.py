import network_kcc
import torch
import onnx
import json

model = torch.load("./word_segmentation_model.pt", map_location="cpu")

with open("tokenizers.json", "w") as f:
  json.dump(model.kccs2int, f, ensure_ascii=False, indent=2)

torch.onnx.export(
    model,
    torch.ones(1, 50, dtype=torch.int64),
    "word_segmentation_model.onnx",
    export_params=True,
    verbose=False,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
      "input":{1: "sequence"},
      "output": {0: "sequence"}
    }
)

model_onnx = onnx.load("./word_segmentation_model.onnx")
onnx.checker.check_model(model_onnx)