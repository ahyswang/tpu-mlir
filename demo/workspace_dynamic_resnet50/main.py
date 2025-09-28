import torch
import torchvision
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import random

# ============================================================================================
# Get the path of the current executing file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
data_dir = os.path.join(current_dir)
temp_dir = os.path.join(current_dir)

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

print(f"Current file path: {current_file_path}")
print(f"Current file directory: {current_dir}")
print(f"Current file name: {current_file_name}")
print(f"Data file directory: {data_dir}")
print(f"Temporary file directory: {temp_dir}")

def set_random_seed(seed=42):
    """
    设置所有相关库的随机种子以确保结果可复现
    """
    # Python内置random模块
    random.seed(seed)

    # NumPy随机种子
    np.random.seed(seed)

    # PyTorch随机种子
    torch.manual_seed(seed)

    # 如果使用GPU，还需要设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对于多GPU情况
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_dynamic_shape():

    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc = torch.nn.Linear(16 * 16, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), x.size(1), -1)
            x = self.fc(x)
            return x

    model = MyModule()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch_output = model(dummy_input)

    dummy_input_np = dummy_input.numpy()
    torch_output_np = torch_output.numpy()

    np.savez(temp_dir + "/dynamic_shape_input.npz", **{'input': dummy_input_np})
    np.savez(temp_dir + "/dynamic_shape_output.npz", **{'output': torch_output_np})

    model.eval()
    onnx_model_path = temp_dir + "/dynamic_shape_model.onnx"
    torch.onnx.export(
        model,                  # Model to export
        dummy_input,            # Model input (can be tuple or multiple parameters)
        onnx_model_path,        # Output file name
        export_params=True,     # Export trained parameter weights
        opset_version=11,       # ONNX opset version to use
        do_constant_folding=True, # Whether to execute constant folding optimization
        input_names = ['input'],   # Input tensor names
        output_names = ['output'], # Output tensor names
        dynamic_axes={'input' : {0 : 'batch_size'}, # Set batch dimension as dynamic
                      'output' : {0 : 'batch_size'}}
    )


def test_reset50():
    # 1. Load pretrained ResNet-50 model and set to evaluation mode
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.eval() # Very important! Set to evaluation mode

    # 2. Create a dummy input
    # Dimensions: (batch_size, channels, height, width)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # 3. Specify the exported ONNX file name
    onnx_model_path = temp_dir + "resnet50.onnx"

    # 4. Export model to ONNX format
    torch.onnx.export(
        model,                  # Model to export
        dummy_input,            # Model input (can be tuple or multiple parameters)
        onnx_model_path,        # Output file name
        export_params=True,     # Export trained parameter weights
        opset_version=11,       # ONNX opset version to use
        do_constant_folding=True, # Whether to execute constant folding optimization
        input_names = ['input'],   # Input tensor names
        output_names = ['output'], # Output tensor names
        # dynamic_axes={'input' : {0 : 'batch_size'}, # Set batch dimension as dynamic
        #               'output' : {0 : 'batch_size'}}
    )

    # --- Optional: Validate the exported ONNX model ---

    # 5. Check if the exported ONNX model format is correct
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed!")

    # 6. Use ONNX Runtime for inference to validate results
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Prepare input data (need to convert to numpy array)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    # Run inference
    ort_outs = ort_session.run(None, ort_inputs)

    # 7. Compare PyTorch and ONNX Runtime output results to ensure consistency
    with torch.no_grad():
        torch_output = model(dummy_input)

if __name__ == "__main__":
    set_random_seed()
    test_dynamic_shape()
    #test_reset50()
