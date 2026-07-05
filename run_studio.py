import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import time

pipe = None

def load_pipe():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        pipe.enable_attention_slicing()
    return pipe

def generate(sketch, prompt, steps, cfg):
    if sketch is None:
        return None, "Please upload a sketch first."
    try:
        p     = load_pipe()
        start = time.time()
        img   = p(
            prompt=f"{prompt}, high fashion photography, white background",
            negative_prompt="blurry, low quality, ugly, deformed",
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            height=512, width=512,
        ).images[0]
        return img, f"Done in {time.time()-start:.0f}s"
    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="Fashion AI") as app:
    gr.Markdown("## AI Fashion Sketch2Design")
    with gr.Row():
        with gr.Column():
            sketch = gr.Image(label="Upload sketch", type="numpy", height=300)
            prompt = gr.Textbox(label="Prompt", value="elegant A-line dress, silk fabric, evening wear", lines=2)
            with gr.Row():
                steps = gr.Slider(10, 30, value=10, step=1, label="Steps")
                cfg   = gr.Slider(5, 12, value=7.5, label="CFG")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            out  = gr.Image(label="Generated design", height=300)
            info = gr.Textbox(label="Info")
    btn.click(generate, inputs=[sketch, prompt, steps, cfg], outputs=[out, info])

print("Starting server...")
app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
