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
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

pip install --upgrade-strategy eager optimum[onnxruntime]
pip install Pillow==9.5.0 sshkeyboard diffusers transformers accelerate
pip install spidev RPi.GPIO numba
# sudo raspi-config enable SPI I2C

# RAG related
#https://pimylifeup.com/raspberry-pi-postgresql/
# sudo apt install -y tesseract-ocr poppler-utils
pip install llama-cpp-haystack
pip install fastembed-haystack qdrant-haystack
pip install pypdf
pip install pydantic==1.10.13
# embedding data goes to /home/kevin/llmware_data/accounts/llmware


# llm ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull openhermes

# model
git lfs install
git clone https://huggingface.co/tommy1900/waifu-lcm-onnx ./

# start
chmod +x "./main.py"
$PYTHON_COMMAND ./main.py