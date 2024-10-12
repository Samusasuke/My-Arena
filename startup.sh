#!/bin/bash

# Check if Python and pip are installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found, please install it first."
    exit
fi

if ! command -v pip3 &> /dev/null
then
    echo "pip3 could not be found, please install it first."
    exit
fi

# Create a Python virtual environment
python3 -m venv ArenaEnv

# Activate the virtual environment
source ArenaEnv/bin/activate

# Determine the operating system and install PyTorch accordingly
echo "Please specify your operating system (mac/linux):"
read os

if [ "$os" == "mac" ]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$os" == "linux" ]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Invalid operating system specified. Please run the script again and specify either 'mac' or 'linux'."
    deactivate
    exit 1
fi

# Install other requirements from requirements.txt
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "requirements.txt file not found. Please ensure it exists in the current directory."
fi

if command -v code &> /dev/null
then
    code --install-extension ms-python.python
    code --install-extension ms-toolsai.jupyter
else
    echo "VSCode command 'code' not found. Please ensure VSCode is installed and the 'code' command is available in your PATH."
fi

git config --global user.name "samusasuke"
git config --global user.email "samuelprietolima@gmail.com"

if [ -f ./.secrets ]; then
    source ./secrets
    wandb login --relogin
fi

echo "Setup complete. The Python virtual environment 'ArenaEnv' is ready to use."
