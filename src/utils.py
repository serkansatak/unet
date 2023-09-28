from PIL import Image
import torchvision.transforms.functional as F
from typing import List
import torch
from torchvision import transforms


def save_tensor_images(tensor_list: List[torch.Tensor], save_path):
    """
    Save input, output, and target tensors as a single image.

    Args:
        input_tensor (torch.Tensor): The input image tensor.
        output_tensor (torch.Tensor): The output image tensor.
        target_tensor (torch.Tensor): The target image tensor.
        save_path (str): The file path to save the combined image.
    """
    # Convert tensors to PIL images
    im_list = []
    for im in tensor_list:
        im = F.to_pil_image(im.cpu())
        im_list.append(im)

    # Create a blank canvas to combine the images
    width, height = im_list[0].size
    combined_image = Image.new("RGB", (width * 3, height))

    # Paste input, output, and target images side by side
    for i, im in enumerate(im_list):
        combined_image.paste(im, (i * width, 0))

    # Save the combined image
    combined_image.save(save_path)


def inverse_normalize(output_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # Multiply by standard deviation and add mean
    denormalized_output = output_tensor * std + mean

    return denormalized_output
