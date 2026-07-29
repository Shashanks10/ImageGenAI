# Infrastructure as Code (OpenTofu/Terraform)

This directory contains the OpenTofu configuration to deploy a GPU-accelerated EC2 instance on AWS for training the Peaky Blinders LoRA module.

## 🛠️ Architecture
- **Provider**: AWS
- **Instance Type**: `g4dn.xlarge` (NVIDIA T4 GPU, 16GB VRAM) - selected for optimal cost/performance for LoRA training.
- **Security**: 
  - Port 22 (SSH) enabled for remote access.
  - Port 8000 (FastAPI) enabled for testing the inference API.

## 🚀 How to Deploy

### 1. Prerequisites
- Install [OpenTofu](https://opentofu.org/) (or Terraform).
- Configure your AWS credentials (`aws configure`).
- Create an SSH Key Pair in your AWS console.

### 2. Configuration
Before running, you need to provide your specific AWS values. You can do this via a `terraform.tfvars` file or as command-line arguments.

Create a `terraform.tfvars` file:
```hcl
key_name = "your-ssh-key-name"
ami_id   = "ami-xxxxxxxxxxxx" # Find a Deep Learning AMI GPU in your region
aws_region = "us-east-1"
```

### 3. Execution
```bash
cd infrastructure
tofu init
tofu plan
tofu apply
```

## 📦 Post-Deployment Setup
Once the instance is live:
1. SSH into the server: `ssh -i your-key.pem ubuntu@<instance_public_ip>`
2. Clone the repository: `git clone <repo_url>`
3. Create a virtual environment and install requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run training: `python Train/train_lora.py`
