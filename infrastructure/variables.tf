variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (GPU instance)"
  type        = string
  default     = "g4dn.xlarge" # NVIDIA T4 GPU, 16GB VRAM - Great for LoRA training
}

variable "ami_id" {
  description = "AMI ID for Deep Learning base (Ubuntu 22.04 with CUDA)"
  type        = string
  default = "ami-0b6d9d3d33ba97d99"
  # Example: Deep Learning AMI GPU usually provides drivers pre-installed
  # Users should update this to the latest DLAMI in their region
}
