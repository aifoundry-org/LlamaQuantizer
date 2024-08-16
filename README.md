# LlamaQuantizer

## Automated Serial Quantization for LLaMA Models

**LlamaQuantizer** is a lightweight tool built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp) that simplifies the process of serial quantizing llama.cpp-supported models. With just a few steps, you can create quantized model set like [OLMo-1B-0724-hf-Quantized](https://huggingface.co/aifoundry-org/OLMo-1B-0724-hf-Quantized).


## How it Works
LlamaQuantizer relies on the [llama.cpp](https://github.com/ggerganov/llama.cpp) library to perform serial automated quantization on LLaMA models. The tool takes care of the following steps:

1. Convertion HF-formatted models to GGUF;
2. Calcaulation of importance matrix if neccessary;
3. Quantizing the model weights and activations on multiple precision levels;
4. Saving the quantized model set to the output directory.

## Basic usage
To create a quantized version of the **OLMo-1B-0724-hf** model, follow these steps:

1. Clone this repo with 
```
git clone https://github.com/aifoundry-org/LlamaQuantizer.git
```
2. Make sure that the required modules are present in your env
```
pip install -r ./requirements.txt
```
3. Download and install llamacpp repo according this guide: [Build llama.cpp locally
](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
5. Download huggingface repo like this one: [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf)
4. Run `quantizer.py` with correct paths to your cloned hf model repo with model and installed llamacpp repo:
```
 ./quantizer.py --llamacpp_path ../llama.cpp/ --model_name OLMo-7B-SFT-hf-0724 --model_dir ../OLMo-7B-0724-SFT-hf/
```

## Example with importance matrix
To make 1-bit and some of 2-bit quantizations, you need to calulate and use an *importance matrix* LlamaQuantizer can handle it! This is the example with importance matrix:

```
./quantizer.py --llamacpp_path ../llama.cpp/ --model_name OLMo-7B-SFT-hf-0724 --model_dir ../OLMo-7B-0724-SFT-hf/--imatrix_text_name wikitext --imatrix_data_path ../wikitext/wikitext-2-raw-v1/train-00000-of-00001.txt
```

## Quantized model sets created or updated with this tool:

#### OLMo-1B:
* [OLMo-1B-0724-hf-Quantized](https://huggingface.co/aifoundry-org/OLMo-1B-0724-hf-Quantized)
#### OLMo-7B:
* [OLMo-7B-0724-hf-Quantized](https://huggingface.co/aifoundry-org/OLMo-7B-0724-hf-Quantized)
* [OLMo-7B-0724-Instruct-hf-Quantized](https://huggingface.co/aifoundry-org/OLMo-7B-0724-Instruct-hf-Quantized)
* [OLMo-7B-0724-SFT-hf-Quantized](https://huggingface.co/aifoundry-org/OLMo-7B-0724-SFT-hf-Quantized)
* [OLMo-7B-0424-hf-Quantized](https://huggingface.co/aifoundry-org/OLMo-7B-0424-hf-Quantized)


## Contributing
Contributions to LlamaQuantizer are welcome! If you'd like to report a bug, request a feature, or submit a pull request, please use this repo's tools.

## License
LlamaQuantizer is released under the Apache 2.0.
