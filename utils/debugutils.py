
import torch


def print_tensor_shapes(tensor_dict):
    if not tensor_dict:
        print("输入的字典为空")
        return

    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"键: {key}, 形状: {value.shape}")
        else:
            print(f"键: {key}, 不是张量类型，类型为: {type(value).__name__}")