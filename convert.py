import onnx
from onnx2torch import convert

# Path to ONNX model
onnx_model_path = 'best.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
import torch
torch.save(torch_model_2, "best.pth")
"""
def _returned_path_to_onnx(dir_path):
    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        # Check if the file has the .onnx extension
        if filename.endswith('.onnx'):
            return os.path.join(dir_path, filename)
    return None


def onnx_to_torch(onnx_model, torch_model):
    """'''Convert an ONNX model to a PyTorch model.'''"""
    # Map ONNX operators to their corresponding PyTorch modules
    op_to_module = {
        'Conv': torch.nn.Conv2d,
        'BatchNormalization': torch.nn.BatchNorm2d,
        'Relu': torch.nn.ReLU,
        'MaxPool': torch.nn.MaxPool2d,
        'AveragePool': torch.nn.AvgPool2d,
        'Flatten': torch.nn.Flatten,
        'Linear': torch.nn.Linear,
        'Sigmoid': torch.nn.Sigmoid,
        'Gemm': torch.nn.Linear,  # Gemm is used for fully connected layers in ONNX
    }

    # Create a dictionary to store the PyTorch modules
    modules = {}

    # Recursively create the PyTorch modules from the ONNX model
    for node in onnx_model.graph.node:
        # Get the input and output names of the node
        inputs = []
        for i in node.input:
            if isinstance(i, str):
                # Constant initializer
                inputs.append(onnx_model.graph.initializer[i].data)
            else:
                # Tensor input
                inputs.append(i.name)
        outputs = [o.name for o in node.output]

        # Get the operator type of the node
        op_type = node.op_type

        # Create the corresponding PyTorch module
        if op_type in op_to_module:
            attrs = {attr.name: attr for attr in node.attribute}
            if 'weight' in attrs:
                attrs['weight'] = torch.tensor(attrs['weight'].t().data)
            if 'bias' in attrs:
                attrs['bias'] = torch.tensor(attrs['bias'].data)
            module = op_to_module[op_type](*attrs.get('shape', (1,)), **attrs)
        else:
            raise NotImplementedError(f"Unsupported ONNX operator: {op_type}")

        # Add the module to the dictionary
        modules[outputs[0]] = module

    # Replace the PyTorch model's modules with the ones created from the ONNX model
    for name, module in torch_model.named_modules():
        if name in modules:
            module.data = modules[name].data


def convert_onnx2pt(onnx_path, save_path):
    # Check if the ONNX file exists
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at {onnx_path}")

    # Load the ONNX model
    try:
        onnx_model = onnx.load("best.onnx")
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX file: {e}")

    # Convert the ONNX model to a PyTorch model
    try:
        torch_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        onnx_to_torch(onnx_model, torch_model)
    except Exception as e:
        raise RuntimeError(f"Failed to convert ONNX model to PyTorch model: {e}")

    # Evaluate the PyTorch model
    torch_model.eval()

    # Export the PyTorch model to a .pt file
    torch.save(torch_model.state_dict(), os.path.join(save_path, "model.pt"))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert an ONNX model to a PyTorch model')
    parser.add_argument('--onnx-path', type=str, default='./outputs',
                        help='Path to the directory containing the ONNX model')
    parser.add_argument('--save-path', type=str, default='.',
                        help='Path to the directory where the PyTorch model will be saved')
    args = parser.parse_args()

    # Convert the ONNX model to a PyTorch model
    convert_onnx2pt(args.onnx_path, args.save_path)

    # Print a success message
    print('Conversion successful!')


if __name__ == '__main__':
    main()
"""