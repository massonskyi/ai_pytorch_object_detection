import argparse
import os
import torch
import torchvision


def _returned_path_to_onnx(dir_path):
    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        # Check if the file has the .onnx extension
        if filename.endswith('.onnx'):
            return os.path.join(dir_path, filename)
    return None


def convert_onnx2pt(onnx_path, save_path):
    # Load the ONNX model
    model_path = _returned_path_to_onnx(onnx_path)
    if model_path is None:
        raise FileNotFoundError('No ONNX file found in the specified directory')
    onnx_model = torch.jit.load(model_path)

    # Create a PyTorch model with the same architecture as the ONNX model
    torch_model = torchvision.models.resnet50(pretrained=False)

    # Load the state dictionary of the PyTorch model from the ONNX model
    torch_model.load_state_dict(onnx_model.state_dict())

    # Evaluate the PyTorch model
    torch_model.eval()

    # Export the PyTorch model to a .pt file
    torch.save(torch_model.state_dict(), os.path.join(save_path, 'model.pt'))


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
