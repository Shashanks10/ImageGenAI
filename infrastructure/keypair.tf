resource "tls_private_key" "ml_key" {
  algorithm = "ED25519"
}

resource "local_file" "private_key" {
  filename        = "${path.module}/futuregenimage.pem"
  content         = tls_private_key.ml_key.private_key_openssh
  file_permission = "0400"
}

resource "aws_key_pair" "ml_key" {
  key_name   = "futuregenimage-key"
  public_key = tls_private_key.ml_key.public_key_openssh
}