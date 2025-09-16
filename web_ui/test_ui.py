#!/usr/bin/env python3
"""
Test version of SkyReels V2 Web UI - for testing the interface without heavy dependencies
"""

import gradio as gr
import os
import time
import random

def mock_text_to_video(prompt, model_id, resolution, num_frames, guidance_scale, 
                      shift, inference_steps, fps, seed, use_prompt_enhancer, progress=gr.Progress()):
    """Mock text-to-video generation for testing."""
    progress(0.1, desc="Initializing...")
    time.sleep(1)
    
    progress(0.5, desc="Generating video...")
    time.sleep(2)
    
    progress(0.9, desc="Saving video...")
    time.sleep(1)
    
    progress(1.0, desc="Complete!")
    
    # Return mock message since we can't generate real videos without models
    return f"Mock video generation completed!\n\nPrompt: {prompt}\nModel: {model_id}\nResolution: {resolution}\nFrames: {num_frames}\nSeed: {seed or 'random'}"

def mock_image_to_video(image, prompt, model_id, resolution, num_frames, 
                       guidance_scale, shift, inference_steps, fps, seed, progress=gr.Progress()):
    """Mock image-to-video generation for testing."""
    if image is None:
        return "Error: Please upload an image."
    
    progress(0.1, desc="Processing image...")
    time.sleep(1)
    
    progress(0.5, desc="Generating video...")
    time.sleep(2)
    
    progress(1.0, desc="Complete!")
    
    return f"Mock I2V generation completed!\n\nImage uploaded: ✅\nPrompt: {prompt}\nModel: {model_id}\nResolution: {resolution}"

def mock_diffusion_forcing(prompt, model_id, resolution, num_frames, base_num_frames,
                          ar_step, overlap_history, addnoise_condition, guidance_scale,
                          shift, inference_steps, fps, seed, image, end_image, progress=gr.Progress()):
    """Mock diffusion forcing generation for testing."""
    progress(0.1, desc="Initializing Diffusion Forcing...")
    time.sleep(1)
    
    progress(0.3, desc="Loading model...")
    time.sleep(1)
    
    progress(0.7, desc="Generating long video...")
    time.sleep(3)
    
    progress(1.0, desc="Complete!")
    
    duration = num_frames / fps
    return f"Mock Diffusion Forcing completed!\n\nPrompt: {prompt}\nTotal frames: {num_frames}\nEstimated duration: {duration:.1f}s\nAR Step: {ar_step}\nModel: {model_id}"

# Model configurations for the dropdowns
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

def create_test_interface():
    """Create the test Gradio interface."""
    
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
    """
    
    with gr.Blocks(css=custom_css, title="SkyReels V2 - Test Interface") as demo:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1 style="margin: 0; font-size: 2.5em;">🎬 SkyReels V2 (Test Mode)</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2em;">Infinite-Length Film Generative Model</p>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Testing Interface - Mock Video Generation</p>
            </div>
        """)
        
        with gr.Tabs():
            # Text-to-Video Tab
            with gr.TabItem("📝 Text-to-Video"):
                gr.HTML('<div class="feature-card"><h3>🎯 Text-to-Video Generation (Test)</h3><p>Test interface for text-to-video generation.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(
                            label="Video Description",
                            placeholder="Describe the video you want to create...",
                            lines=3,
                            value="A serene lake surrounded by towering mountains, with swans gliding across the water."
                        )
                        
                        with gr.Row():
                            t2v_model = gr.Dropdown(
                                choices=MODEL_CONFIGS["text2video"],
                                label="Model",
                                value=MODEL_CONFIGS["text2video"][0]
                            )
                            t2v_resolution = gr.Dropdown(
                                choices=["540P", "720P"],
                                label="Resolution", 
                                value="540P"
                            )
                        
                        with gr.Row():
                            t2v_num_frames = gr.Slider(49, 121, 97, label="Number of Frames")
                            t2v_fps = gr.Slider(8, 30, 24, label="FPS")
                        
                        with gr.Row():
                            t2v_guidance_scale = gr.Slider(1.0, 15.0, 6.0, label="Guidance Scale")
                            t2v_shift = gr.Slider(1.0, 12.0, 8.0, label="Flow Shift")
                        
                        with gr.Row():
                            t2v_inference_steps = gr.Slider(10, 100, 30, label="Inference Steps")
                            t2v_seed = gr.Number(label="Seed (optional)", precision=0, value=None)
                        
                        t2v_use_enhancer = gr.Checkbox(label="Use Prompt Enhancer", value=False)
                        t2v_generate_btn = gr.Button("🎬 Generate Video (Test)", variant="primary")
                    
                    with gr.Column(scale=1):
                        t2v_output = gr.Textbox(
                            label="Generation Result", 
                            lines=10,
                            interactive=False
                        )
                
                t2v_generate_btn.click(
                    fn=mock_text_to_video,
                    inputs=[t2v_prompt, t2v_model, t2v_resolution, t2v_num_frames,
                           t2v_guidance_scale, t2v_shift, t2v_inference_steps, t2v_fps,
                           t2v_seed, t2v_use_enhancer],
                    outputs=t2v_output
                )
            
            # Image-to-Video Tab
            with gr.TabItem("🖼️ Image-to-Video"):
                gr.HTML('<div class="feature-card"><h3>🎨 Image-to-Video Generation (Test)</h3><p>Test interface for image-to-video generation.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(label="Input Image", type="pil", height=200)
                        i2v_prompt = gr.Textbox(
                            label="Video Description",
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
                                choices=["540P", "720P"],
                                label="Resolution",
                                value="540P"
                            )
                        
                        with gr.Row():
                            i2v_num_frames = gr.Slider(49, 121, 97, label="Number of Frames")
                            i2v_fps = gr.Slider(8, 30, 24, label="FPS")
                        
                        with gr.Row():
                            i2v_guidance_scale = gr.Slider(1.0, 15.0, 5.0, label="Guidance Scale")
                            i2v_shift = gr.Slider(1.0, 12.0, 3.0, label="Flow Shift")
                        
                        with gr.Row():
                            i2v_inference_steps = gr.Slider(10, 100, 30, label="Inference Steps")
                            i2v_seed = gr.Number(label="Seed (optional)", precision=0, value=None)
                        
                        i2v_generate_btn = gr.Button("🎬 Generate Video (Test)", variant="primary")
                    
                    with gr.Column(scale=1):
                        i2v_output = gr.Textbox(
                            label="Generation Result",
                            lines=10, 
                            interactive=False
                        )
                
                i2v_generate_btn.click(
                    fn=mock_image_to_video,
                    inputs=[i2v_image, i2v_prompt, i2v_model, i2v_resolution,
                           i2v_num_frames, i2v_guidance_scale, i2v_shift, i2v_inference_steps,
                           i2v_fps, i2v_seed],
                    outputs=i2v_output
                )
            
            # Diffusion Forcing Tab
            with gr.TabItem("🎞️ Long Video (Diffusion Forcing)"):
                gr.HTML('<div class="feature-card"><h3>🚀 Infinite-Length Video Generation (Test)</h3><p>Test interface for diffusion forcing long video generation.</p></div>')
                
                with gr.Row():
                    with gr.Column(scale=1):
                        df_prompt = gr.Textbox(
                            label="Video Description",
                            lines=3,
                            value="A graceful white swan swimming in a serene lake at dawn."
                        )
                        
                        with gr.Row():
                            df_start_image = gr.Image(label="Start Frame (optional)", type="pil", height=120)
                            df_end_image = gr.Image(label="End Frame (optional)", type="pil", height=120)
                        
                        with gr.Row():
                            df_model = gr.Dropdown(
                                choices=MODEL_CONFIGS["diffusion_forcing"],
                                label="Model",
                                value=MODEL_CONFIGS["diffusion_forcing"][1]
                            )
                            df_resolution = gr.Dropdown(
                                choices=["540P", "720P"],
                                label="Resolution",
                                value="540P"
                            )
                        
                        with gr.Row():
                            df_num_frames = gr.Slider(97, 1457, 257, label="Total Frames")
                            df_base_frames = gr.Slider(57, 121, 97, label="Base Frames")
                        
                        with gr.Row():
                            df_ar_step = gr.Slider(0, 10, 0, label="AR Step")
                            df_overlap_history = gr.Slider(17, 37, 17, label="Overlap History")
                        
                        with gr.Row():
                            df_addnoise = gr.Slider(0, 50, 20, label="Add Noise Condition")
                            df_guidance_scale = gr.Slider(1.0, 15.0, 6.0, label="Guidance Scale")
                        
                        with gr.Row():
                            df_shift = gr.Slider(1.0, 12.0, 8.0, label="Flow Shift")
                            df_inference_steps = gr.Slider(10, 100, 30, label="Inference Steps")
                        
                        with gr.Row():
                            df_fps = gr.Slider(8, 30, 24, label="FPS")
                            df_seed = gr.Number(label="Seed (optional)", precision=0, value=None)
                        
                        df_generate_btn = gr.Button("🎬 Generate Long Video (Test)", variant="primary")
                    
                    with gr.Column(scale=1):
                        df_output = gr.Textbox(
                            label="Generation Result",
                            lines=10,
                            interactive=False
                        )
                
                df_generate_btn.click(
                    fn=mock_diffusion_forcing,
                    inputs=[df_prompt, df_model, df_resolution, df_num_frames, df_base_frames,
                           df_ar_step, df_overlap_history, df_addnoise, df_guidance_scale,
                           df_shift, df_inference_steps, df_fps, df_seed, df_start_image, df_end_image],
                    outputs=df_output
                )
            
            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🌟 SkyReels V2 Test Interface</h3>
                    <p>This is a test version of the SkyReels V2 web interface. It demonstrates the UI functionality 
                    without requiring the full model dependencies.</p>
                    
                    <h4>✅ Test Features</h4>
                    <ul>
                        <li>✅ Interface Layout and Navigation</li>
                        <li>✅ Parameter Controls and Validation</li>
                        <li>✅ Progress Tracking</li>
                        <li>✅ Mock Generation Pipeline</li>
                        <li>⚠️ Actual video generation requires full model setup</li>
                    </ul>
                    
                    <h4>🚀 Next Steps</h4>
                    <p>To use the full functionality:</p>
                    <ol>
                        <li>Install all dependencies: <code>pip install -r requirements.txt</code></li>
                        <li>Download the SkyReels V2 models from Hugging Face</li>
                        <li>Run the full interface: <code>python launch.py</code></li>
                    </ol>
                </div>
                """)
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <p>🧪 <strong>SkyReels V2 Test Interface</strong> - UI Testing Mode</p>
                <p style="font-size: 0.9em; color: #666;">Interface validation complete - Ready for full deployment!</p>
            </div>
        """)
    
    return demo

if __name__ == "__main__":
    print("🧪 SkyReels V2 Test Interface")
    print("=" * 40)
    print("✅ Testing UI without heavy dependencies")
    print("🚀 Starting test interface...")
    print("📱 Available at: http://localhost:7860")
    
    demo = create_test_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )