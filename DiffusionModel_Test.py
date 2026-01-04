import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image

# 1. SETUP
FACE_IMAGE_PATH = "test_face.jpg"
OUT_IMAGE = "test_face.png"

# 2. LOAD & PREPARE IMAGE
# Resize to square (1024x1024) to match SDXL training data
face_image = Image.open(FACE_IMAGE_PATH).convert("RGB").resize((1024, 1024))

# 3. LOAD PIPELINE
base_model = "cagliostrolab/animagine-xl-3.1"
ip_adapter_repo = "h94/IP-Adapter"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False
)

# 4. LOAD IP-ADAPTER
# FIX: Use the standard 'ip-adapter_sdxl.bin' (Matches the pipeline's 1280-dim encoder)
pipe.load_ip_adapter(
    ip_adapter_repo,
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

# 5. OPTIMIZATIONS (Must be AFTER loading adapter)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# 6. SCHEDULER (SDE Karras is best for Animagine)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++"
)

# 7. GENERATE
prompt = (
    "masterpiece, best quality, very aesthetic, absurdres, "
    " 1girl, smiling, beautiful"
    "studio ghibli style, cell shaded"
)

negative_prompt = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, "
    "blurry, realistic, photo, 3d render, lips, nose, "
    "fused bodies, fused heads, multiple boys, multiple girls, mutation"
)

# Scale: 0.6 is a balanced starting point
pipe.set_ip_adapter_scale(0.6)

print("Generating...")
image = pipe(
    prompt=prompt,
    ip_adapter_image=face_image,
    guidance_scale=5.0,
    num_inference_steps=30,
    negative_prompt=negative_prompt,
).images[0]

image.save(OUT_IMAGE)
print(f"Saved to {OUT_IMAGE}")