output "instance_public_ip" {
  description = "Public IP address of the GPU training server"
  value       = aws_instance.training_server.public_ip
}

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.training_server.id
}
