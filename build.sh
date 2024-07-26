#!/bin/bash

# Read the model ID from the file

LOCAL_MODEL_DIR="./local_model_dir"
GGUF_MODEL_DIR="./models/mymodel"
CUDA_SUPPORT=1
QUANT_METHOD="Q4_K_M"
MODEL_FILE_DIR="D:/model"

# Install necessary dependencies
pip install -i https://pypi.org/simple/ bitsandbytes
pip install peft transformers trl datasets accelerate ollama make cmake 
pip install -r requirements.txt

# Run the Python finetuning script
echo "Running the Python finetuning script..."
python finetuning_script.py
MODEL_ID=$(cat model_id.txt)
# Download the Hugging Face Model
echo "Downloading the Hugging Face model..."
huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_MODEL_DIR"

# Clone the llama.cpp Repository
echo "Cloning llama.cpp repository..."
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build llama.cpp
echo "Building llama.cpp with CUDA support..."
git pull
make clean
LLAMA_CUDA=$CUDA_SUPPORT make

# Install Required Python Packages for llama.cpp
echo "Installing required Python packages..."
pip install -r requirements.txt

# Build llama.cpp Using CMake
echo "Building llama.cpp project using CMake..."
cmake . -B build
cmake --build build --config Release 

# Convert the Model
echo "Converting the Hugging Face model to a GGUF file..."
python convert_hf_to_gguf.py "$LOCAL_MODEL_DIR"

# Move GGUF model to a distinct directory
mkdir -p "$GGUF_MODEL_DIR"
mv "$LOCAL_MODEL_DIR"/ggml-model-f16.gguf "$GGUF_MODEL_DIR"

# Quantize the Model to 4-bit
echo "Quantizing the model to 4-bit..."
./llama-quantize "$GGUF_MODEL_DIR"/ggml-model-f16.gguf "$GGUF_MODEL_DIR"/ggml-model-"$QUANT_METHOD".gguf "$QUANT_METHOD"

# Create a Modelfile
echo "Creating the modelfile..."
MODEL_NAME=$(basename "$MODEL_FILE_DIR")
echo "FROM $MODEL_FILE_DIR/ggml-model-$QUANT_METHOD.gguf" > "$MODEL_FILE_DIR/$MODEL_NAME.Modelfile"
echo "SYSTEM Answer questions in detail." >> "$MODEL_FILE_DIR/$MODEL_NAME.Modelfile"

# Create a Model using ollama
echo "Creating model using ollama..."
ollama create -f "$MODEL_NAME" -f "$MODEL_FILE_DIR/$MODEL_NAME.Modelfile"

# Run the Model
echo "Running the model..."
ollama run  -f "$MODEL_NAME"
