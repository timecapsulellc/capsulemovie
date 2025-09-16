# SkyReels V2 Web UI

A comprehensive web interface for the SkyReels V2 infinite-length film generative model, providing easy access to all video generation and analysis capabilities.

## 🌟 Features

### 🎬 Video Generation
- **Text-to-Video**: Generate videos from text descriptions
- **Image-to-Video**: Animate static images with AI
- **Diffusion Forcing**: Create infinite-length videos (10s, 30s, 60s+)
- **Frame Control**: Control start and end frames
- **Prompt Enhancement**: Automatically improve text prompts

### 🎯 Video Analysis
- **SkyCaptioner**: Generate detailed structural captions for videos
- **Multi-format Support**: Analyze various video formats

### ⚙️ Advanced Controls
- **Multiple Models**: Choose from 1.3B, 5B, and 14B parameter models
- **Resolution Options**: 540P and 720P generation
- **Precision Controls**: Fine-tune all generation parameters
- **Real-time Progress**: Track generation progress

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SkyworkAI/SkyReels-V2
   cd SkyReels-V2
   ```

2. **Install dependencies**:
   ```bash
   cd web_ui
   pip install -r requirements.txt
   ```

3. **Launch the web interface**:
   ```bash
   python launch.py
   ```

4. **Access the interface**:
   - Local: http://localhost:7860
   - Network: http://0.0.0.0:7860

## 📋 Usage Guide

### Text-to-Video Generation

1. Navigate to the **Text-to-Video** tab
2. Enter your video description in the prompt field
3. Select your preferred model and resolution
4. Adjust generation parameters:
   - **Number of Frames**: 49-121 (affects video length)
   - **Guidance Scale**: 1.0-15.0 (higher = more prompt adherence)
   - **Flow Shift**: 1.0-12.0 (8.0 recommended for T2V)
   - **Inference Steps**: 10-100 (higher = better quality, slower)
5. Optionally enable prompt enhancement
6. Click **Generate Video**

### Image-to-Video Generation

1. Navigate to the **Image-to-Video** tab
2. Upload your source image
3. Describe how the image should animate
4. Select appropriate I2V model
5. Adjust parameters:
   - **Guidance Scale**: 5.0 recommended for I2V
   - **Flow Shift**: 3.0 recommended for I2V
6. Click **Generate Video**

### Long Video Generation (Diffusion Forcing)

1. Navigate to the **Long Video** tab
2. Enter your video description
3. Optionally provide start/end frame images
4. Configure long video parameters:
   - **Total Frames**: 257 (10s), 737 (30s), 1457 (60s)
   - **Base Frames**: 97 (affects memory usage)
   - **AR Step**: 0 for sync, >0 for async generation
   - **Overlap History**: 17-37 (for smooth transitions)
5. Click **Generate Long Video**

### Video Captioning

1. Navigate to the **Video Captioning** tab
2. Upload your video file
3. Click **Generate Caption**
4. View the detailed JSON structure analysis

## 🎛️ Parameter Guide

### Core Parameters

| Parameter | Range | Description | Recommended |
|-----------|-------|-------------|-------------|
| **Resolution** | 540P/720P | Output resolution | 540P for speed, 720P for quality |
| **Num Frames** | 49-121 | Video length in frames | 97 (4s @ 24fps) |
| **Guidance Scale** | 1.0-15.0 | Text adherence strength | 6.0 (T2V), 5.0 (I2V) |
| **Flow Shift** | 1.0-12.0 | Generation flow parameter | 8.0 (T2V), 3.0 (I2V) |
| **Inference Steps** | 10-100 | Quality vs speed trade-off | 30-50 |

### Diffusion Forcing Parameters

| Parameter | Range | Description | Recommended |
|-----------|-------|-------------|-------------|
| **AR Step** | 0-10 | Async steps (0=synchronous) | 0 for quality, 5 for long videos |
| **Base Frames** | 57-121 | Memory-quality balance | 97 (540P), 121 (720P) |
| **Overlap History** | 17-37 | Frame overlap for continuity | 17 |
| **Add Noise** | 0-50 | Consistency enhancement | 20 |

## 🔧 Advanced Configuration

### Model Selection

**Text-to-Video Models:**
- `Skywork/SkyReels-V2-T2V-14B-540P` - High quality, 540P
- `Skywork/SkyReels-V2-T2V-14B-720P` - Highest quality, 720P

**Image-to-Video Models:**
- `Skywork/SkyReels-V2-I2V-1.3B-540P` - Fast, lower VRAM
- `Skywork/SkyReels-V2-I2V-14B-540P` - Balanced performance
- `Skywork/SkyReels-V2-I2V-14B-720P` - Highest quality

**Diffusion Forcing Models:**
- `Skywork/SkyReels-V2-DF-1.3B-540P` - Memory efficient
- `Skywork/SkyReels-V2-DF-14B-540P` - Balanced performance
- `Skywork/SkyReels-V2-DF-14B-720P` - Highest quality

### Memory Requirements

| Model Size | Resolution | VRAM Required | Generation Time |
|------------|------------|---------------|-----------------|
| 1.3B | 540P | ~15GB | Fast |
| 14B | 540P | ~45GB | Medium |
| 14B | 720P | ~55GB | Slow |

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `base_num_frames` parameter
   - Use smaller model (1.3B instead of 14B)
   - Lower resolution (540P instead of 720P)
   - Enable model offloading

2. **Generation Too Slow**:
   - Reduce `inference_steps`
   - Use TeaCache acceleration (if available)
   - Try async generation with lower AR steps

3. **Poor Quality Results**:
   - Increase `inference_steps`
   - Improve prompt quality
   - Use prompt enhancer
   - Try different guidance scale values

4. **Model Download Issues**:
   - Check internet connection
   - Ensure sufficient disk space
   - Verify Hugging Face access

### Performance Tips

- **For Long Videos**: Use asynchronous generation (ar_step > 0)
- **For Quality**: Higher inference steps and guidance scale
- **For Speed**: Lower resolution and inference steps
- **For Memory**: Use model offloading and smaller base frames

## 📁 Output Files

Generated videos are saved in:
```
result/web_ui/
├── t2v_[seed]_[timestamp].mp4    # Text-to-video outputs
├── i2v_[seed]_[timestamp].mp4    # Image-to-video outputs
└── df_[seed]_[timestamp].mp4     # Diffusion forcing outputs
```

## 🔗 Links & Resources

- **📄 Technical Report**: [arXiv:2504.13074](https://arxiv.org/pdf/2504.13074)
- **🌐 Official Website**: [skyreels.ai](https://www.skyreels.ai/home)
- **🤗 Hugging Face**: [SkyReels V2 Collection](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9)
- **💬 Discord**: [Community Support](https://discord.gg/PwM6NYtccQ)
- **📚 Documentation**: [GitHub Repository](https://github.com/SkyworkAI/SkyReels-V2)

## 🤝 Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## 📄 License

This project follows the license terms of the main SkyReels V2 repository.

---

**🎬 SkyReels V2 Web UI** - Bringing infinite-length film generation to your browser!