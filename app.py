import os
import cv2
import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F
from gradio.themes.utils import sizes
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
CHECKPOINTS_DIR = os.path.join(ASSETS_DIR, "checkpoints")
CHECKPOINTS = {
    "0.3b": "sapiens_0.3b_render_people_epoch_100_torchscript.pt2",
    "0.6b": "sapiens_0.6b_render_people_epoch_70_torchscript.pt2",
    "1b": "sapiens_1b_render_people_epoch_88_torchscript.pt2",
    "2b": "sapiens_2b_render_people_epoch_25_torchscript.pt2",
}
SEG_CHECKPOINT = 'sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2'

def load_model(checkpoint_name: str):
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    model = torch.jit.load(checkpoint_path)
    model.eval()
    model.to("cuda")
    return model

MODELS = {name: load_model(CHECKPOINTS[name]) for name in CHECKPOINTS.keys()}
SEG_MODEL = load_model(SEG_CHECKPOINT)

@torch.inference_mode()
def run_model(model, input_tensor, height, width):
    output = model(input_tensor)
    output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    return output

transform_fn = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
])

@spaces.GPU
def process_image(image: Image.Image, model_name: str) -> Image.Image:
    input_tensor = transform_fn(image).unsqueeze(0).to("cuda")
    
    # Run segmentation
    seg_output = run_model(SEG_MODEL, input_tensor, image.height, image.width)
    seg_mask = (seg_output.argmax(dim=1) > 0).float().cpu().numpy()[0]

    # Run depth estimation
    depth_model = MODELS[model_name]
    depth_output = run_model(depth_model, input_tensor, image.height, image.width)
    depth_map = depth_output.squeeze().cpu().numpy()

    # Apply segmentation mask to depth map
    depth_map[seg_mask == 0] = np.nan

    # Normalize and colorize depth map
    depth_foreground = depth_map[~np.isnan(depth_map)]
    if len(depth_foreground) > 0:
        min_val, max_val = np.nanmin(depth_foreground), np.nanmax(depth_foreground)
        depth_normalized = (depth_map - min_val) / (max_val - min_val)
        depth_normalized = 1 - depth_normalized  # Invert the depth values
        depth_normalized = np.nan_to_num(depth_normalized, nan=0)
        
        # Use matplotlib's colormap instead of cv2
        cmap = plt.get_cmap('inferno')
        depth_colored = (cmap(depth_normalized) * 255).astype(np.uint8)[:, :, :3]
    else:
        depth_colored = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    return Image.fromarray(depth_colored)

with open("header.html", "r") as file:
    header = file.read()

CUSTOM_CSS = """
.image-container img {
    max-width: 1024px;
    max-height: 512px;
    margin: 0 auto;
    border-radius: 0px;
}
.gradio-container {background-color: #000000}
"""

js_func = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(js=js_func, css=CUSTOM_CSS, theme=gr.themes.Monochrome(radius_size=sizes.radius_md)) as demo:
    gr.HTML(header)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil", format="png")
            model_name = gr.Dropdown(
                label="Model Size",
                choices=list(CHECKPOINTS.keys()),
                value="0.3b",
            )
            example_model = gr.Examples(
                inputs=input_image,
                examples_per_page=14,
                examples=[
                    os.path.join(ASSETS_DIR, "images", img)
                    for img in os.listdir(os.path.join(ASSETS_DIR, "images"))
                ],
            )
        with gr.Column():
            result_image = gr.Image(label="Depth Estimation Result", format="png")
            run_button = gr.Button("Run")

    run_button.click(
        fn=process_image,
        inputs=[input_image, model_name],
        outputs=[result_image],
    )

if __name__ == "__main__":
    demo.launch(share=False)