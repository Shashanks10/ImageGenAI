variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (GPU instance)"
  type        = string
  default     = "g5.2xlarge" # NVIDIA A10G GPU, 24GB VRAM, 32GB RAM - Required for FLUX.1 LoRA training
}

variable "ami_id" {
  description = "AMI ID for Deep Learning base (Ubuntu 22.04 with CUDA)"
  type        = string
  default = "ami-0b6d9d3d33ba97d99"
  # Example: Deep Learning AMI GPU usually provides drivers pre-installed
  # Users should update this to the latest DLAMI in their region
}
