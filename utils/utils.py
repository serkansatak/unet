from PIL import Image
import torchvision.transforms.functional as F


def save_tensor_images(input_tensor, output_tensor, target_tensor, save_path):
    """
    Save input, output, and target tensors as a single image.

    Args:
        input_tensor (torch.Tensor): The input image tensor.
        output_tensor (torch.Tensor): The output image tensor.
        target_tensor (torch.Tensor): The target image tensor.
        save_path (str): The file path to save the combined image.
    """
    # Convert tensors to PIL images
    input_image = F.to_pil_image(input_tensor.cpu())
    output_image = F.to_pil_image(output_tensor.cpu())
    target_image = F.to_pil_image(target_tensor.cpu())

    # Create a blank canvas to combine the images
    width, height = input_image.size
    combined_image = Image.new("RGB", (width * 3, height))

    # Paste input, output, and target images side by side
    combined_image.paste(input_image, (0, 0))
    combined_image.paste(output_image, (width, 0))
    combined_image.paste(target_image, (2 * width, 0))

    # Save the combined image
    combined_image.save(save_path)
