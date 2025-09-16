#!/usr/bin/env python3
"""
SkyReels V2 Comprehensive Web UI
A unified interface for all video generation and captioning functionality
"""

import gradio as gr
import os
import sys
import json
import time
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "Capsule Movie"))

try:
    from skyreels_v2_infer.modules import download_model
    from skyreels_v2_infer.pipelines import Text2VideoPipeline, Image2VideoPipeline, PromptEnhancer
    from skyreels_v2_infer.pipelines import resizecrop
    from skyreels_v2_infer import DiffusionForcingPipeline
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SkyReels modules: {e}")
    MODELS_AVAILABLE = False

# Configuration
SUPPORTED_RESOLUTIONS = ["540P", "720P"]
MODEL_CONFIGS = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
    "diffusion_forcing": [
        "Skywork/SkyReels-V2-DF-1.3B-540P",
        "Skywork/SkyReels-V2-DF-14B-540P",
        "Skywork/SkyReels-V2-DF-14B-720P",
    ]
}

class SkyReelsWebUI:
    def __init__(self):
        self.current_pipeline = None
        self.pipeline_type = None
        
    def get_resolution_dims(self, resolution: str) -> Tuple[int, int]:
        """Get height and width for resolution."""
        if resolution == "540P":
            return 544, 960
        elif resolution == "720P":
            return 720, 1280
        else:
            raise ValueError(f"Invalid resolution: {resolution}")
    
    def text_to_video(self, prompt: str, model_id: str, resolution: str, 
                     num_frames: int, guidance_scale: float, shift: float,
                     inference_steps: int, fps: int, seed: Optional[int],
                     use_prompt_enhancer: bool, progress=gr.Progress()) -> str:
        """Generate video from text prompt."""
        if not MODELS_AVAILABLE:
            return "Error: SkyReels models not available. Please install required dependencies."
        
        try:
            progress(0.1, desc="Initializing...")
            
            # Set random seed if not provided
            if seed is None:
                seed = random.randint(0, 4294967294)
            
            # Get resolution dimensions
            height, width = self.get_resolution_dims(resolution)
            
            # Initialize prompt enhancer if requested
            enhanced_prompt = prompt
            if use_prompt_enhancer:
                progress(0.2, desc="Enhancing prompt...")
                try:
                    prompt_enhancer = PromptEnhancer()
                    enhanced_prompt = prompt_enhancer(prompt)
                    print(f"Enhanced prompt: {enhanced_prompt}")
                except Exception as e:
                    print(f"Prompt enhancement failed: {e}")
            
            progress(0.3, desc="Loading model...")
            
            # Download and initialize model
            model_path = download_model(model_id)
            pipe = Text2VideoPipeline(
                model_path=model_path, 
                dit_path=model_path, 
                use_usp=False, 
                offload=True
            )
            
            progress(0.5, desc="Generating video...")
            
            # Generate video
            import torch
            import imageio
            
            kwargs = {
                "prompt": enhanced_prompt,
                "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality",
                "num_frames": num_frames,
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "generator": torch.Generator(device="cuda").manual_seed(seed),
                "height": height,
                "width": width,
            }
            
            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                video_frames = pipe(**kwargs)[0]
            
            progress(0.9, desc="Saving video...")
            
            # Save video
            os.makedirs("result/web_ui", exist_ok=True)
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            video_filename = f"t2v_{seed}_{current_time}.mp4"
            output_path = os.path.join("result/web_ui", video_filename)
            
            imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
            
            progress(1.0, desc="Complete!")
            return output_path
            
        except Exception as e:
            return f"Error generating video: {str(e)}"
    
    def image_to_video(self, image, prompt: str, model_id: str, resolution: str,
                      num_frames: int, guidance_scale: float, shift: float,
                      inference_steps: int, fps: int, seed: Optional[int],
                      progress=gr.Progress()) -> str:
        """Generate video from image and prompt."""
        if not MODELS_AVAILABLE:
            return "Error: SkyReels models not available. Please install required dependencies."
        
        if image is None:
            return "Error: Please upload an image."
        
        try:
            progress(0.1, desc="Initializing...")
            
            # Set random seed if not provided
            if seed is None:
                seed = random.randint(0, 4294967294)
            
            # Get resolution dimensions
            height, width = self.get_resolution_dims(resolution)
            
            progress(0.3, desc="Loading model...")
            
            # Download and initialize model
            model_path = download_model(model_id)
            pipe = Image2VideoPipeline(
                model_path=model_path,
                dit_path=model_path,
                use_usp=False,
                offload=True
            )
            
            # Process image
            from PIL import Image
            if isinstance(image, str):
                input_image = Image.open(image).convert("RGB")
            else:
                input_image = image.convert("RGB")
            
            # Resize and crop image
            image_width, image_height = input_image.size
            if image_height > image_width:
                height, width = width, height
            input_image = resizecrop(input_image, height, width)
            
            progress(0.5, desc="Generating video...")
            
            # Generate video
            import torch
            import imageio
            
            kwargs = {
                "prompt": prompt,
                "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality",
                "image": input_image,
                "num_frames": num_frames,
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "generator": torch.Generator(device="cuda").manual_seed(seed),
                "height": height,
                "width": width,
            }
            
            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                video_frames = pipe(**kwargs)[0]
            
            progress(0.9, desc="Saving video...")
            
            # Save video
            os.makedirs("result/web_ui", exist_ok=True)
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            video_filename = f"i2v_{seed}_{current_time}.mp4"
            output_path = os.path.join("result/web_ui", video_filename)
            
            imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
            
            progress(1.0, desc="Complete!")
            return output_path
            
        except Exception as e:
            return f"Error generating video: {str(e)}"
    
    def diffusion_forcing_video(self, prompt: str, model_id: str, resolution: str,
                               num_frames: int, base_num_frames: int, ar_step: int,
                               overlap_history: Optional[int], addnoise_condition: int,
                               guidance_scale: float, shift: float, inference_steps: int,
                               fps: int, seed: Optional[int], image=None, end_image=None,
                               progress=gr.Progress()) -> str:
        """Generate long video using Diffusion Forcing."""
        if not MODELS_AVAILABLE:
            return "Error: SkyReels models not available. Please install required dependencies."
        
        try:
            progress(0.1, desc="Initializing Diffusion Forcing...")
            
            # Set random seed if not provided
            if seed is None:
                seed = random.randint(0, 4294967294)
            
            # Get resolution dimensions
            height, width = self.get_resolution_dims(resolution)
            
            # Validation
            if num_frames > base_num_frames and overlap_history is None:
                overlap_history = 17  # Default recommended value
            
            progress(0.3, desc="Loading Diffusion Forcing model...")
            
            # Download and initialize model
            import torch
            model_path = download_model(model_id)
            pipe = DiffusionForcingPipeline(
                model_path,
                dit_path=model_path,
                device=torch.device("cuda"),
                weight_dtype=torch.bfloat16,
                use_usp=False,
                offload=True,
            )
            
            # Process images if provided
            start_image = None
            final_image = None
            
            if image is not None:
                from PIL import Image
                if isinstance(image, str):
                    start_image = Image.open(image).convert("RGB")
                else:
                    start_image = image.convert("RGB")
                
                image_width, image_height = start_image.size
                if image_height > image_width:
                    height, width = width, height
                start_image = resizecrop(start_image, height, width)
            
            if end_image is not None:
                from PIL import Image
                if isinstance(end_image, str):
                    final_image = Image.open(end_image).convert("RGB")
                else:
                    final_image = end_image.convert("RGB")
                final_image = resizecrop(final_image, height, width)
            
            progress(0.5, desc="Generating long video...")
            
            # Generate video using Diffusion Forcing
            import imageio
            
            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
                    image=start_image,
                    end_image=final_image,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=inference_steps,
                    shift=shift,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    overlap_history=overlap_history,
                    addnoise_condition=addnoise_condition,
                    base_num_frames=base_num_frames,
                    ar_step=ar_step,
                    causal_block_size=5 if ar_step > 0 else 1,
                    fps=fps,
                )[0]
            
            progress(0.9, desc="Saving long video...")
            
            # Save video
            os.makedirs("result/web_ui", exist_ok=True)
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            video_filename = f"df_{seed}_{current_time}.mp4"
            output_path = os.path.join("result/web_ui", video_filename)
            
            imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
            
            progress(1.0, desc="Complete!")
            return output_path
            
        except Exception as e:
            return f"Error generating long video: {str(e)}"

# Initialize the web UI
skyreels_ui = SkyReelsWebUI()

def create_interface():
    """Create the main Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .feature-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .gradio-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    """
    
    with gr.Blocks(css=custom_css, title="SkyReels V2 - Infinite Film Generation") as demo:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.5em;">🎬 SkyReels V2</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2em;">Infinite-Length Film Generative Model</p>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Advanced AI Video Generation with Diffusion Forcing Technology</p>
            </div>
        """)
        
        with gr.Tabs():
            # Text-to-Video Tab
            with gr.TabItem("📝 Text-to-Video", elem_id="t2v_tab"):
                gr.HTML('<div class="feature-card"><h3>🎯 Text-to-Video Generation</h3><p>Create videos from text descriptions using state-of-the-art AI models.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(
                            label="Video Description",
                            placeholder="Describe the video you want to create...",
                            lines=3,
                            value="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface."
                        )
                        
                        with gr.Row():
                            t2v_model = gr.Dropdown(
                                choices=MODEL_CONFIGS["text2video"],
                                label="Model",
                                value=MODEL_CONFIGS["text2video"][0]
                            )
                            t2v_resolution = gr.Dropdown(
                                choices=SUPPORTED_RESOLUTIONS,
                                label="Resolution",
                                value="540P"
                            )
                        
                        with gr.Row():
                            t2v_num_frames = gr.Slider(
                                minimum=49, maximum=121, step=24, value=97,
                                label="Number of Frames"
                            )
                            t2v_fps = gr.Slider(
                                minimum=8, maximum=30, step=1, value=24,
                                label="FPS"
                            )
                        
                        with gr.Row():
                            t2v_guidance_scale = gr.Slider(
                                minimum=1.0, maximum=15.0, step=0.5, value=6.0,
                                label="Guidance Scale"
                            )
                            t2v_shift = gr.Slider(
                                minimum=1.0, maximum=12.0, step=0.5, value=8.0,
                                label="Flow Shift"
                            )
                        
                        with gr.Row():
                            t2v_inference_steps = gr.Slider(
                                minimum=10, maximum=100, step=5, value=30,
                                label="Inference Steps"
                            )
                            t2v_seed = gr.Number(
                                label="Seed (optional)",
                                precision=0,
                                value=None
                            )
                        
                        t2v_use_enhancer = gr.Checkbox(
                            label="Use Prompt Enhancer",
                            value=False
                        )
                        
                        t2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary")
                    
                    with gr.Column(scale=1):
                        t2v_output = gr.Video(
                            label="Generated Video",
                            height=400
                        )
                        t2v_info = gr.HTML(label="Generation Info")
                
                # Connect T2V functionality
                t2v_generate_btn.click(
                    fn=skyreels_ui.text_to_video,
                    inputs=[t2v_prompt, t2v_model, t2v_resolution, t2v_num_frames,
                           t2v_guidance_scale, t2v_shift, t2v_inference_steps, t2v_fps,
                           t2v_seed, t2v_use_enhancer],
                    outputs=t2v_output
                )
            
            # Image-to-Video Tab
            with gr.TabItem("🖼️ Image-to-Video", elem_id="i2v_tab"):
                gr.HTML('<div class="feature-card"><h3>🎨 Image-to-Video Generation</h3><p>Animate static images with AI-powered video generation.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=300
                        )
                        i2v_prompt = gr.Textbox(
                            label="Video Description",
                            placeholder="Describe how the image should be animated...",
                            lines=3,
                            value="The image comes to life with gentle movements and natural motion."
                        )
                        
                        with gr.Row():
                            i2v_model = gr.Dropdown(
                                choices=MODEL_CONFIGS["image2video"],
                                label="Model",
                                value=MODEL_CONFIGS["image2video"][-1]
                            )
                            i2v_resolution = gr.Dropdown(
                                choices=SUPPORTED_RESOLUTIONS,
                                label="Resolution",
                                value="540P"
                            )
                        
                        with gr.Row():
                            i2v_num_frames = gr.Slider(
                                minimum=49, maximum=121, step=24, value=97,
                                label="Number of Frames"
                            )
                            i2v_fps = gr.Slider(
                                minimum=8, maximum=30, step=1, value=24,
                                label="FPS"
                            )
                        
                        with gr.Row():
                            i2v_guidance_scale = gr.Slider(
                                minimum=1.0, maximum=15.0, step=0.5, value=5.0,
                                label="Guidance Scale"
                            )
                            i2v_shift = gr.Slider(
                                minimum=1.0, maximum=12.0, step=0.5, value=3.0,
                                label="Flow Shift"
                            )
                        
                        with gr.Row():
                            i2v_inference_steps = gr.Slider(
                                minimum=10, maximum=100, step=5, value=30,
                                label="Inference Steps"
                            )
                            i2v_seed = gr.Number(
                                label="Seed (optional)",
                                precision=0,
                                value=None
                            )
                        
                        i2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary")
                    
                    with gr.Column(scale=1):
                        i2v_output = gr.Video(
                            label="Generated Video",
                            height=400
                        )
                        i2v_info = gr.HTML(label="Generation Info")
                
                # Connect I2V functionality
                i2v_generate_btn.click(
                    fn=skyreels_ui.image_to_video,
                    inputs=[i2v_image, i2v_prompt, i2v_model, i2v_resolution,
                           i2v_num_frames, i2v_guidance_scale, i2v_shift, i2v_inference_steps,
                           i2v_fps, i2v_seed],
                    outputs=i2v_output
                )
            
            # Diffusion Forcing Tab (Long Videos)
            with gr.TabItem("🎞️ Long Video (Diffusion Forcing)", elem_id="df_tab"):
                gr.HTML('<div class="feature-card"><h3>🚀 Infinite-Length Video Generation</h3><p>Create long-form videos using cutting-edge Diffusion Forcing technology.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        df_prompt = gr.Textbox(
                            label="Video Description",
                            placeholder="Describe the long video sequence...",
                            lines=3,
                            value="A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface."
                        )
                        
                        with gr.Row():
                            df_start_image = gr.Image(
                                label="Start Frame (optional)",
                                type="pil",
                                height=150
                            )
                            df_end_image = gr.Image(
                                label="End Frame (optional)",
                                type="pil",
                                height=150
                            )
                        
                        with gr.Row():
                            df_model = gr.Dropdown(
                                choices=MODEL_CONFIGS["diffusion_forcing"],
                                label="Model",
                                value=MODEL_CONFIGS["diffusion_forcing"][1]
                            )
                            df_resolution = gr.Dropdown(
                                choices=SUPPORTED_RESOLUTIONS,
                                label="Resolution",
                                value="540P"
                            )
                        
                        with gr.Row():
                            df_num_frames = gr.Slider(
                                minimum=97, maximum=1457, step=80, value=257,
                                label="Total Frames (257=10s, 737=30s)"
                            )
                            df_base_frames = gr.Slider(
                                minimum=57, maximum=121, step=20, value=97,
                                label="Base Frames"
                            )
                        
                        with gr.Row():
                            df_ar_step = gr.Slider(
                                minimum=0, maximum=10, step=1, value=0,
                                label="AR Step (0=sync, >0=async)"
                            )
                            df_overlap_history = gr.Slider(
                                minimum=17, maximum=37, step=10, value=17,
                                label="Overlap History"
                            )
                        
                        with gr.Row():
                            df_addnoise = gr.Slider(
                                minimum=0, maximum=50, step=5, value=20,
                                label="Add Noise Condition"
                            )
                            df_guidance_scale = gr.Slider(
                                minimum=1.0, maximum=15.0, step=0.5, value=6.0,
                                label="Guidance Scale"
                            )
                        
                        with gr.Row():
                            df_shift = gr.Slider(
                                minimum=1.0, maximum=12.0, step=0.5, value=8.0,
                                label="Flow Shift"
                            )
                            df_inference_steps = gr.Slider(
                                minimum=10, maximum=100, step=5, value=30,
                                label="Inference Steps"
                            )
                        
                        with gr.Row():
                            df_fps = gr.Slider(
                                minimum=8, maximum=30, step=1, value=24,
                                label="FPS"
                            )
                            df_seed = gr.Number(
                                label="Seed (optional)",
                                precision=0,
                                value=None
                            )
                        
                        df_generate_btn = gr.Button("🎬 Generate Long Video", variant="primary")
                    
                    with gr.Column(scale=1):
                        df_output = gr.Video(
                            label="Generated Long Video",
                            height=400
                        )
                        df_info = gr.HTML(label="Generation Info")
                
                # Connect DF functionality
                df_generate_btn.click(
                    fn=skyreels_ui.diffusion_forcing_video,
                    inputs=[df_prompt, df_model, df_resolution, df_num_frames, df_base_frames,
                           df_ar_step, df_overlap_history, df_addnoise, df_guidance_scale,
                           df_shift, df_inference_steps, df_fps, df_seed, df_start_image, df_end_image],
                    outputs=df_output
                )
            
            # Video Captioning Tab
            with gr.TabItem("📹 Video Captioning", elem_id="caption_tab"):
                gr.HTML('<div class="feature-card"><h3>🎤 SkyCaptioner - Video Analysis</h3><p>Generate detailed structural captions for videos using AI.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        caption_video = gr.Video(
                            label="Upload Video for Analysis",
                            height=300
                        )
                        caption_btn = gr.Button("🎯 Generate Caption", variant="primary")
                        
                        gr.HTML("""
                        <p><strong>Note:</strong> Video captioning requires the SkyCaptioner model to be installed and configured.</p>
                        """)
                    
                    with gr.Column(scale=1):
                        caption_output = gr.Code(
                            label="Generated Caption (JSON)",
                            language="json",
                            lines=20,
                            interactive=False
                        )
            
            # About Tab
            with gr.TabItem("ℹ️ About", elem_id="about_tab"):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🌟 About SkyReels V2</h3>
                    <p>SkyReels V2 is the world's first infinite-length film generative model using a Diffusion Forcing framework. 
                    This comprehensive web interface provides access to all the advanced video generation capabilities.</p>
                    
                    <h4>✨ Features</h4>
                    <ul>
                        <li><strong>Text-to-Video:</strong> Generate videos from text descriptions</li>
                        <li><strong>Image-to-Video:</strong> Animate static images with AI</li>
                        <li><strong>Diffusion Forcing:</strong> Create infinite-length videos</li>
                        <li><strong>Video Captioning:</strong> Analyze and describe video content</li>
                        <li><strong>Prompt Enhancement:</strong> Automatically improve prompts</li>
                        <li><strong>Frame Control:</strong> Control start and end frames</li>
                    </ul>
                    
                    <h4>🏆 Performance</h4>
                    <p>SkyReels V2 achieves state-of-the-art performance among publicly available models, 
                    with superior instruction adherence and visual quality.</p>
                    
                    <h4>🔗 Links</h4>
                    <ul>
                        <li><a href="https://arxiv.org/pdf/2504.13074" target="_blank">📄 Technical Report</a></li>
                        <li><a href="https://www.skyreels.ai/home" target="_blank">🌐 Official Website</a></li>
                        <li><a href="https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9" target="_blank">🤗 Hugging Face</a></li>
                        <li><a href="https://discord.gg/PwM6NYtccQ" target="_blank">💬 Discord Community</a></li>
                    </ul>
                </div>
                """)
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <p>🎬 <strong>SkyReels V2</strong> - Infinite-Length Film Generative Model</p>
                <p style="font-size: 0.9em; color: #666;">Powered by Diffusion Forcing Technology | © 2025 Skywork AI</p>
            </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )