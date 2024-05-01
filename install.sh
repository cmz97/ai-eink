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
# BLAS follow this https://github.com/OpenMathLib/OpenBLAS/wiki/Faq#replacing-system-blasupdating-apt-openblas-in-mintubuntudebian
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make DYNAMIC_ARCH=1
sudo make DYNAMIC_ARCH=1 install
sudo apt install libblas-dev liblapack-dev
sudo update-alternatives --install /usr/lib/libblas.so.3 libblas.so.3 /opt/OpenBLAS/lib/libopenblas.so.0 41 \
   --slave /usr/lib/liblapack.so.3 liblapack.so.3 /opt/OpenBLAS/lib/libopenblas.so.0
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

# pip instals
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


# Whisper
pip install pydub
git clone https://github.com/ggerganov/whisper.cpp.git 
cd whisper.cpp/
bash ./models/download-ggml-model.sh tiny.en-q5_1
OPENBLAS_PATH=/opt/OpenBLAS WHISPER_OPENBLAS=1 make libwhisper.so -j
# to run 
LD_LIBRARY_PATH=/opt/OpenBLAS WHISPER_OPENBLAS=1 python mic_script.py 

# alternatve
pip install faster-whisper
# translation 



# github auth 
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# llm ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull openhermes

# model
git lfs install
git clone https://huggingface.co/tommy1900/waifu-lcm-onnx ./

# start
chmod +x "./main.py"
$PYTHON_COMMAND ./main.py