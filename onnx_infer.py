from onnxruntime import InferenceSession

session = InferenceSession("./word_segmentation_model.onnx")
output = session.run(None, {"input": [[692, 1238,    2,  862,  484, 1561,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1, 1742,  484, 1448, 2966, 1716, 3604,  484, 3710,
         2079, 1574, 3013,    3, 1336,    1,    1,    1,    1,    1,    1,    1]]})

result = output[0]
result[result >= 0.5] = 1.0
result[result < 0.5] = 0.0
print(result)


