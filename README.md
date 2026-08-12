<div align="center">

# 🎨 FutureGenAI — Next-Gen AI Image Generation Engine

<p align="center">
  <b>High-Performance FLUX.1 & Stable Diffusion Engine with 4-Bit LoRA Fine-Tuning, Img2Img Transformation, FastAPI, Docker & AWS Infrastructure</b>
</p>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Diffusers%20%26%20PEFT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Terraform](https://img.shields.io/badge/Terraform-AWS%20EC2-7B42BC?style=for-the-badge&logo=terraform&logoColor=white)](https://www.terraform.io/)

</div>

---

## 📌 Overview

**FutureGenAI** is an end-to-end, production-grade AI image generation platform. Built on top of state-of-the-art **FLUX.1-dev**, **FLUX.2-dev**, and **Stable Diffusion**, it seamlessly handles:

1. **🎨 Cool Poster & Stylized Art Generation** (LoRA style adapter with automatic `cool_style` trigger).
2. **🖼️ Image-to-Image (Img2Img) Transformation** (Convert standard photos into artistic vector posters).
3. **✨ Pure Text-to-Image (T2I)** (Generate stylized or photo-realistic images directly from prompts).
4. **⚡ VRAM Optimization** (4-bit NF4 Transformer quantization via `bitsandbytes`, reducing VRAM consumption from ~24GB down to **~6GB**).
5. **🧠 Custom LoRA Fine-Tuning** (Train your own custom styles on local image-caption datasets or Hugging Face Hub datasets).
6. **🚀 Production REST API & Infrastructure** (FastAPI service, Dockerized containerization, and automated AWS EC2 GPU provisioning with Terraform).

---

## 🚀 Key Features

- **⚡ 4-Bit NF4 Quantization**: Loads FLUX.1 Transformer in 4-bit `nf4` mode with double quantization, enabling high-quality FLUX generation even on consumer GPUs (e.g., RTX 3060 / 4060).
- **🎭 Multi-Modal Endpoints**:
  - `POST /generate-poster`: Upload an image for Img2Img style transfer OR omit for T2I poster art.
  - `POST /generate-text`: Direct T2I poster generation with custom prompts and dimensions.
  - `POST /generate-normal`: Base image generation using FLUX.2-dev without LoRA styling.
- **📦 End-to-End Fine-Tuning**: PyTorch Dataset loaders with online data augmentation (Random Resized Crop, Flips, Color Jitter) and `peft` LoRA weight extraction.
- **🐳 Docker Containerization**: Multi-stage Linux environment with NVIDIA CUDA 12.1 runtime, non-root user execution, and automated health checks.
- **☁️ Cloud Infrastructure**: Infrastructure as Code (IaC) with Terraform for automated deployment of GPU servers on AWS EC2.

---

## 🏗️ System Architecture & Data Flow

```
                      ┌───────────────────────────────────────────┐
                      │              Dataset Source               │
                      │  (Local Images/Captions OR HuggingFace)   │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │         Dataset Loader (dataset.py)       │
                      │      Data Augmentation & Tokenization     │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │    FLUX.1 / SD Model + LoRA Adapter      │
                      │               (model.py)                  │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │       Training Loop (train_lora.py)       │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │          Save LoRA Weights (.safetensors) │
                      └─────────────────────┬─────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │   Inference Engine        │                 │       FastAPI Web API     │
       │   (generate.py / normal)  │                 │    (router.py / service)  │
       └───────────────────────────┘                 └─────────────┬─────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │ User/Client  │
                                                            └──────────────┘
```

---

## 📂 Project Structure

```bash
ImageGenAI/
├── Api/                      # FastAPI REST Web Service
│   ├── main.py               # FastAPI entrypoint & router initialization
│   ├── router.py             # API route definitions (/generate-poster, /generate-text, /generate-normal)
│   └── service.py            # Service wrapper bridging API endpoints with inference engine
│
├── Inference/                # Model Inference Engine
│   ├── generate.py           # FLUX.1 PosterGenerator with 4-bit NF4 & LoRA support
│   ├── generate_normal.py    # NormalGenerator (FLUX.2-dev standard T2I pipeline)
│   └── load_lora.py          # Helper for loading LoRA weights into diffusers pipeline
│
├── Train/                    # Fine-Tuning & Model Training
│   ├── train_lora.py         # Complete LoRA training script (PyTorch + PEFT)
│   ├── dataset.py            # PyTorch Dataset loaders (Local directory & HuggingFace Hub)
│   ├── model.py              # Model loader & LoRA injection module
│   └── output/               # Saved LoRA checkpoints & safetensors
│
├── infrastructure/           # Cloud Deployment (Terraform)
│   ├── main.tf               # AWS EC2 GPU instance & Security Group configuration
│   ├── keypair.tf            # Keypair provisioning
│   ├── providers.tf          # Terraform AWS provider settings
│   └── variables.tf          # Configurable variables (AMI, Instance Type, Region)
│
├── Dockerfile                # Production NVIDIA CUDA 12.1 Docker container
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Hardware Requirements
- **NVIDIA GPU**: Minimum 8GB VRAM (12GB+ recommended for faster inference/training).
- **RAM**: 16GB+ System Memory.
- **CUDA**: 12.1 compatible driver installed.

### 2. Hugging Face Access Token
`FLUX.1-dev` is a gated model. You must create an access token with permission to read gated models on Hugging Face:
1. Accept the model license at [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).
2. Set your token as an environment variable:

```bash
# On Linux / macOS
export HF_TOKEN="hf_your_huggingface_access_token"

# On Windows (PowerShell)
$env:HF_TOKEN="hf_your_huggingface_access_token"

# On Windows (CMD)
set HF_TOKEN=hf_your_huggingface_access_token
```

---

## ⚡ Quick Start

### Method 1: Local Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Shashanks10/ImageGenAI.git
   cd ImageGenAI
   ```

2. **Create & Activate Virtual Environment**
   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. **Launch the FastAPI Server**
   ```bash
   python Api/main.py
   ```
   *The API server will start at `http://localhost:8000`. Interactive Swagger UI available at `http://localhost:8000/docs`.*

---

### Method 2: Docker Setup

Run the GPU-accelerated container using Docker with NVIDIA Container Toolkit:

1. **Build Docker Image**
   ```bash
   docker build -t futuregenimage:latest .
   ```

2. **Run Docker Container**
   ```bash
   docker run -d \
     --gpus all \
     -p 8000:8000 \
     -e HF_TOKEN="hf_your_huggingface_access_token" \
     --name futuregenimage_app \
     futuregenimage:latest
   ```

---

## 🌐 API Reference

### 1. Root Status
`GET /`
Check API status and available endpoints.

### 2. Generate Cool Poster (Image-to-Image or Text-to-Image)
`POST /generate-poster`
Converts an input image into a stylish vector poster or generates poster art from a prompt.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File (UploadFile)` | No | Image file to transform. Omit for pure Text-to-Image. |
| `prompt` | `String` | No | Prompt describing the output style or concept. |
| `strength` | `Float` | No | Denoising strength for Img2Img (Default: `0.85`). |

**Example (cURL - Image-to-Image):**
```bash
curl -X POST "http://localhost:8000/generate-poster" \
  -F "file=@/path/to/photo.jpg" \
  -F "prompt=cyberpunk warrior hero" \
  -F "strength=0.85" \
  --output poster_result.jpg
```

---

### 3. Generate Poster from Text Prompt
`POST /generate-text`
Generates stylized poster artwork directly from a text prompt.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `String` | Yes | Description of desired artwork. |
| `width` | `Integer` | No | Output image width (Default: `512`). |
| `height` | `Integer` | No | Output image height (Default: `512`). |

**Example (Python):**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-text",
    data={"prompt": "a samurai standing on a neon street at night", "width": 512, "height": 512}
)

with open("samurai_poster.jpg", "wb") as f:
    f.write(response.content)
```

---

### 4. Generate Standard Base Image (No LoRA)
`POST /generate-normal`
Generates clean images using FLUX.2-dev without custom style LoRA adapters.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | `String` | Yes | Standard text prompt. |
| `width` | `Integer` | No | Output width (Default: `512`). |
| `height` | `Integer` | No | Output height (Default: `512`). |

---

## 🏋️ LoRA Fine-Tuning Guide

You can train your own LoRA adapters using `Train/train_lora.py`.

### Local Dataset Format
Prepare your dataset on disk:
```
Dataset/
├── images/
│   ├── 001.jpg
│   └── 002.jpg
└── captions/
    ├── 001.txt
    └── 002.txt
```

### Run Fine-Tuning
```bash
python Train/train_lora.py \
  --image_dir Dataset/images \
  --caption_dir Dataset/captions \
  --output_dir Train/output/my_custom_lora \
  --epochs 100 \
  --lr 1e-4 \
  --batch_size 1
```

*Or train directly from a Hugging Face Hub Dataset:*
```bash
python Train/train_lora.py \
  --hf_repo AIGCDuckBoss/fluxlora_cool-posters \
  --output_dir Train/output/cool_posters_lora \
  --epochs 50 \
  --augment
```

---

## ☁️ Cloud Infrastructure Deployment (AWS Terraform)

Provision a dedicated GPU EC2 server on AWS:

1. **Navigate to Infrastructure Directory**
   ```bash
   cd infrastructure
   ```

2. **Initialize & Apply Terraform**
   ```bash
   terraform init
   terraform plan
   terraform apply -auto-approve
   ```

3. **Outputs**
   Terraform will output the Public IP and Instance ID of your GPU server:
   ```bash
   public_ip = "x.x.x.x"
   instance_id = "i-0xxxxxxxxxxxxxx"
   ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Shashanks10/ImageGenAI/issues).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
  <sub>Built with ❤️ by Shashank & FutureGenAI Team</sub>
</div>
