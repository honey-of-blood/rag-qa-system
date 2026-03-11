#!/bin/bash

# Phase 10: Docker Installation
echo "Starting Docker installation..."

sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo "Installation complete. Please run 'newgrp docker' to apply group changes."
echo "Then verify with:"
echo "docker --version"
echo "docker compose version"
