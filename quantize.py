import torch


def quantize(
        model: torch.nn.Module
) -> torch.nn.Module:
    """
    Quantize the model. This is done by replacing the `torch.nn.quantized module` with `torch.nn.quantized.dynamic`
    and then re-registering the module with the `torch.nn.quantized.dynamic` module
    :param model: The model to quantize the model with the `torch.nn.quantized` module replacing the
    `torch.nn.quantized` module with `torch.nn.quantized.dynamic`
    """

    quantized_model = torch.quantization.convert(model)  # type: torch.nn.Module
    return quantized_model
