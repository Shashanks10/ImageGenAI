provider "aws" {
  region  = var.aws_region
  profile = "test-playground"
}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }

    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }

    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }

#   backend "s3" {
#     bucket = "futuregen-tfstate-bucket"
#     key = "training-server/terraform.tfstate"
#     region = "us-east-1"
#   }
}