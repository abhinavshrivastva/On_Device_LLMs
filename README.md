# Hugging Face Model to GGUF File Conversion

This guide explains how to convert a Hugging Face model to a GGUF file. Follow the steps below to complete the process.

## Prerequisites

- Python installed on your machine
- Git installed on your machine
- `huggingface-cli` installed
- `pip` installed

## Steps

1. **Download the Hugging Face Model**

   Use the `huggingface-cli` to download your desired model from Hugging Face.
    ```bash
   huggingface-cli download -m <Model> --local-dir <local_model_dir>
   ```
   Replace `<Model>` with the name of the Hugging Face model you want to download and `<local_model_dir>` with the directory where you want to save the model.

2. **Clone the llama.cpp Repository**

   Clone the llama.cpp repository from GitHub to get the necessary scripts and files.
    ```bash
   git clone https://github.com/ggerganov/llama.cpp
   ```

3. **Build llama.cpp**

   Navigate to the llama.cpp directory, pull the latest changes, clean previous builds, and then build the project with CUDA support.
    ```bash
   cd llama.cpp && git pull && make clean && LLAMA_CUDA=1 make
   ```

4. **Install Required Python Packages**

   Install the required Python packages listed in the requirements.txt file within the llama.cpp directory.
    ```bash
   pip install -r llama.cpp/requirements.txt
   ```

5. **Convert the Model**

   Finally, run the conversion script to convert the downloaded Hugging Face model to a GGUF file.
    ```bash
   python llama.cpp/convert_hf_to_gguf.py <local_model_dir>
   ```
   Replace `<local_model_dir>` with the directory where you saved the Hugging Face model.

By following these steps, you should be able to successfully convert a Hugging Face model to a GGUF file.
