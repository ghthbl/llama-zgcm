# Community ZGCM-1-7B support

This branch ports [ZGCAGI's ZGCM-1-7B](https://huggingface.co/zgcagi/ZGCM-1-7B) to llama.cpp. The model was trained by ZGCAGI; this fork provides conversion and inference support, not a new trained model. This is independent community work, not an official release or endorsement.

[GGUF files, importance matrix, checksums and evaluation](https://huggingface.co/0xtb/ZGCM-1-7B-GGUF). Start with Q6_K; Q8_0, Q5_K_M and an F16/BF16 reference are also available.

Based on llama.cpp commit 41abbfd599fbdd3470fcae0a1fb6530ad8403cd7. The source delta registers the architecture, tokenizer settings and model graph, adds a strict HF converter, and supplies a reference-payload verifier. There are no new GGML operations, Metal kernels or cache implementations.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j 8 --target llama-server llama-cli llama-quantize llama-imatrix
hf download 0xtb/ZGCM-1-7B-GGUF ZGCM-1-7B-BF16-down28-31-IMATRIX-Q6_K.gguf --local-dir .
./build/bin/llama-server -m ZGCM-1-7B-BF16-down28-31-IMATRIX-Q6_K.gguf -c 131072 -ngl 99 -fa on -np 1 --jinja --host 127.0.0.1 --port 8080
```

The model supports context up to 262144. Set -c explicitly. Its existing chat template supports thinking and direct-response modes. Raw text completion and reliable agent tool calling are not recommended as validated use cases. The model can collapse into repeated exclamation marks, including with original BF16 weights; read the model card before use.

Conversion requires a complete local original HF snapshot and the normal convert_hf_to_gguf.py dependencies:

```bash
python convert_hf_to_gguf.py /path/to/ZGCM-1-7B --outtype f16 --outfile ZGCM-1-7B-BF16-downall32-F16.gguf
python scripts/verify-zgcm-gguf.py ZGCM-1-7B-BF16-downall32-F16.gguf --hf-model /path/to/ZGCM-1-7B
```

The converter retains every FFN down-projection as BF16. Quantization must preserve layers 28-31 as BF16; use the model card's explicit tensor override. The imatrix is calibration data and is hosted with the weights, not in this source repository.

Validated primarily on M2 Max with Metal, plus short CPU checks. Other backends are untested. Generated SVG examples show that numerical conversion fidelity does not establish useful output quality on every task.

AI assistants contributed substantially to the implementation, evaluation tooling and documentation under the publisher's direction. The publisher is responsible for the released code. This branch is not an upstream pull request. The original ggml authors' MIT license remains in LICENSE; the model's separate MIT notice is in the Hugging Face repository.
