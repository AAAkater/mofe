from unet.transform import load_model, process_single_image

model_path = "./checkpoints/unet_final.pth"
model, model_device = load_model(model_path)


def image_to_image(file_content: bytes):
    new_image = process_single_image(file_content, model, device=model_device)
    return new_image
