# Security Group for SSH and API Access
resource "aws_security_group" "ml_sg" {
  name        = "futuregenimage-ml-sg"
  description = "Allow SSH and FastAPI"

  # SSH Access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # WARNING: In production, restrict this to your IP
  }

  # FastAPI access
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound allow all
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "futuregenimage-sg"
  }
}

# EC2 Instance for GPU Training
resource "aws_instance" "training_server" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name = aws_key_pair.ml_key.key_name

  # vcpus = 4 # Required for some GPU instances

  vpc_security_group_ids = [aws_security_group.ml_sg.id]

  # Basic User Data to prepare the system
  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y git python3-pip
              echo "FutureGenImage training server is ready."
              EOF

  tags = {
    Name = "FutureGenImage-GPU-Server"
  }

  depends_on = [aws_key_pair.ml_key]
}
