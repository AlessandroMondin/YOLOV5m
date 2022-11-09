import os
import torch
import cv2
import numpy as np

# Tensor.element_size() → int
# Returns the size in bytes of an individual element.
import config


def check_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))

def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def strip_model(model):
    model.half()
    for p in model.parameters():
        p.requires_grid = False

# Tensor.element_size() → int
# Returns the size in bytes of an individual element.
def save_model(model, folder_path, file_name):
    ckpt = {}
    ckpt["model"] = model
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("Saving Model...")
    torch.save(ckpt, os.path.join(folder_path, file_name))

def export_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    input_names = ["actual_input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      dummy_input,
                      "netron_onnx_files/yolov5m_mine.onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      opset_version=11
                      )


def save_checkpoint(state, folder_path, filename, epoch):
    path = os.path.join(folder_path, filename)
    if not os.path.exists(path):
        os.makedirs(path)

    print("=> Saving checkpoint...")
    torch.save(state, os.path.join(path, f"checkpoint_epoch_{str(epoch)}.pth.tar"))


def load_model_checkpoint(model_name, model):
    folder = os.listdir(os.path.join("SAVED_CHECKPOINT", model_name))
    file = folder[-1]
    print(f"=> loading the last epoch of SAVED_CHECKPOINT/{model_name}")

    checkpoint = torch.load(os.path.join("SAVED_CHECKPOINT", model_name, file), map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])


def load_optim_checkpoint(model_name, optim):
    folder = os.listdir(os.path.join("SAVED_CHECKPOINT", model_name))
    file = folder[-1]
    print(f"=> loading the last epoch of SAVED_CHECKPOINT/{model_name}")

    checkpoint = torch.load(os.path.join("SAVED_CHECKPOINT", model_name, file), map_location=config.DEVICE)
    optim.load_state_dict(checkpoint["optimizer"])


def resize_image(image, output_size):
    # output size is [width, height]
    return cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_LINEAR)

def coco91_2_coco80(label):
    # idx & labels below are not present in MS_COCO
    """11: 'street sign', 25: 'hat', 28: 'shoe', 29: 'eye glasses', 44: 'plate', 65: 'mirror',
    67: 'window', 68: 'desk', 70: 'door', 82: 'blender', 90: 'hairbrush'"""
    if 11 < label <= 25:
        return label - 1
    elif 25 < label <= 28:
        return label - 2
    elif 28 < label <= 29:
        return label - 3
    elif 29 < label <= 44:
        return label - 4
    elif 44 < label <= 65:
        return label - 5
    elif 65 < label <= 67:
        return label - 6
    elif 67 < label <= 68:
        return label - 7
    elif 68 < label <= 70:
        return label - 8
    elif 70 < label <= 82:
        return label - 9
    elif 82 < label <= 90:
        return label - 10
    elif label >= 90:
        return label - 11
    else:
        return label


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
