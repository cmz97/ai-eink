#!/usr/bin/env bash
echo Starting FastSD CPU env installation...
set -e
PYTHON_COMMAND="python"

# requirements : python onnx, optimum, PIL, sshkeyboard
apt-get update --assume-yes

if ! command -v python3 &>/dev/null; then
    if ! command -v python &>/dev/null; then
        echo "Error: Python not found, installing python"
        sudo apt-get install -y --assume-yes build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev libffi-dev
        wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tar.xz
        tar xf Python-3.8.10.tar.xz
        cd Python-3.8.10
        ./configure --enable-optimizations --prefix=/usr
        make
        sudo make altinstall
        cd ..
        sudo rm -r Python-3.8.0
        rm Python-3.8.0.tar.xz
        . ~/.bashrc
        sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8.10 1
    fi
fi

if command -v python &>/dev/null; then
   PYTHON_COMMAND="python"
fi

echo "Found $PYTHON_COMMAND command"

python_version=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}')  
echo "Python version : $python_version"

BASEDIR=$(pwd)
# pip installs
pip install --upgrade-strategy eager optimum[onnxruntime]
pip install PIL sshkeyboard diffusers transformers

# llm ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull openhermes

# model
git lfs install
git clone https://huggingface.co/tommy1900/waifu-lcm-onnx ./

# start
chmod +x "./main.py"
$PYTHON_COMMAND ./main.py