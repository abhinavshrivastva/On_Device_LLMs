# Hugging Face Model to GGUF File Conversion

This guide explains how to convert a Hugging Face model to a GGUF file. Follow the steps below to complete the process.

## Prerequisites

- install ollama
- install make
- install cmake
- `huggingface-cli` installed

## Steps

1. **Download the Hugging Face Model**

   Use the `huggingface-cli` to download your desired model from Hugging Face.
    ```bash
   huggingface-cli download <Model ID from HF> --local-dir <local_model_dir>
   ```
   Replace `<Model ID from HF>` with the name of the Hugging Face model you want to download and `<local_model_dir>` with the directory where you want to save the model.

2. **Clone the llama.cpp Repository**

   Clone the llama.cpp repository from GitHub to get the necessary scripts and files.
    ```bash
   git clone https://github.com/ggerganov/llama.cpp
   ```

3. **Build llama.cpp**

   Navigate to the llama.cpp directory, pull the latest changes, clean previous builds, and then build the project with CUDA support.
    ```bash
   cd llama.cpp
   git pull
   make clean
   LLAMA_CUDA=1 make
   ```

4. **Install Required Python Packages**

   Install the required Python packages listed in the `requirements.txt` file within the llama.cpp directory.
    ```bash
   pip install -r requirements.txt
   ```

5. **Build llama.cpp Using CMake**

   Build the llama.cpp project using CMake.
    ```bash
   cmake . -B build
   cmake --build build --config Release 
   ```

6. **Convert the Model**

   Run the conversion script to convert the downloaded Hugging Face model to a GGUF file.
    ```bash
   python llama.cpp/convert_hf_to_gguf.py <local_model_dir>
   ```
   Replace `<local_model_dir>` with the directory where you saved the Hugging Face model.

7. **Quantize the Model to 4-bit**

   After converting the model, you can quantize it to a 4-bit representation.
    ```bash
   ./llama-quantize ./models/mymodel/ggml-model-f16.gguf ./models/mymodel/ggml-model-Q4_K_M.gguf Q4_K_M
   ```

8. **Create a Modelfile**

   Create a modelfile with the template:
    ```
    FROM D:\model\ggml-model-Q4_K_M.gguf  # (directory of the GGUF file)
    SYSTEM Answer questions in detail.
    ```

9. **Create a Model**

   Use the `ollama` command to create a model from the modelfile:
    ```bash
   ollama create <MODEL_NAME> -f "D:\model\<MODEL_NAME>.Modelfile"
   ```

10. **Run the Model**

    Run the model using the `run` command:
    ```bash
    ollama run <MODEL_NAME>
    ```



### Methodology

- **Dependency Installation**: The script installs all necessary Python packages and system dependencies required for model training and conversion.
  
- **Model Download**: The script downloads the Hugging Face model specified by the `model_id.txt` file and stores it in the local directory.
  
- **Building Tools**: It clones and builds the `llama.cpp` project, enabling CUDA support if specified. This tool is used for quantizing the model.
  
- **Model Conversion**: The Hugging Face model is converted to the GGUF format using the `convert_hf_to_gguf.py` script.
  
- **Quantization**: The converted GGUF model is quantized to a 4-bit format to reduce its size and improve performance.
  
- **Model File Creation**: A `Modelfile` is created to define the model configuration for `ollama`.
  
- **Model Creation**: The script uses `ollama` to create and set up the model based on the `Modelfile`.
  
- **Running the Model**: Finally, the model is run using `ollama`.
