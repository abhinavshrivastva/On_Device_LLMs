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


## Running the Pipeline

### Making the Script Executable

Before running the `build.sh` script, you need to ensure it has executable permissions. This allows the script to be run directly from the command line. To do this, use the `chmod` command:

```bash
chmod +x build.sh
```

This command changes the permissions of the `build.sh` file, making it executable. The `+x` flag adds execute permissions for the file.

### Running the Script

Once the script is executable, you can run it using the following command:

```bash
./build.sh
```

This command executes the `build.sh` script, which performs the entire pipeline process:

1. **Installs Dependencies**: The script installs all necessary Python packages and system dependencies required for model fine-tuning and conversion.
   
2. **Downloads the Model**: It retrieves the Hugging Face model specified in `model_id.txt` and saves it to the local directory.
   
3. **Clones and Builds `llama.cpp`**: The script clones the `llama.cpp` repository, builds it with CUDA support, and installs required Python packages.
   
4. **Converts the Model**: It converts the downloaded Hugging Face model to GGUF format.
   
5. **Quantizes the Model**: The script quantizes the GGUF model to a 4-bit format to optimize it.
   
6. **Creates a Modelfile**: It generates a `Modelfile` for use with `ollama`.
   
7. **Sets Up the Model with `ollama`**: Finally, the script creates and configures the model using `ollama` and runs it.

