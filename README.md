# llama-zgcm

**An independent community fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for [ZGCAGI's ZGCM-1-7B](https://huggingface.co/zgcagi/ZGCM-1-7B).**

> This is **not the official llama.cpp repository** and is not affiliated with or endorsed by the llama.cpp maintainers or ZGCAGI. For the official project, releases and general-purpose documentation, visit **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**.

This fork makes ZGCM-1-7B available for local inference through llama.cpp. The original model was trained by ZGCAGI. This project contributes the conversion, runtime support and local evaluation; it does not claim authorship of the model and does not contain a fine-tune.

## Downloads and documentation

- **[GGUF models and importance matrix on Hugging Face](https://huggingface.co/0xtb/ZGCM-1-7B-GGUF)**: Q8_0, Q6_K, Q5_K_M, the mixed F16/BF16 reference, checksums, calibration details and evaluation results. Start with Q6_K for a balance of size and fidelity.
- **[This fork's releases](https://github.com/ghthbl/llama-zgcm/releases)**: tagged source releases for ZGCM support.
- **[ZGCM build, conversion and usage notes](docs/ZGCM.md)**: the implementation scope and commands.
- **[Original model](https://huggingface.co/zgcagi/ZGCM-1-7B)** and **[authors' source repository](https://github.com/zgcagi/ZGCM-1)**: model architecture, training and original documentation.

The weights and imatrix are hosted on Hugging Face, not in this source repository.

## What this fork adds

Native ZGCM architecture registration, its model graph and tokenizer mapping, a strict Hugging Face-to-GGUF converter, and a reference-payload verifier. The implementation uses llama.cpp's existing GGML operations and hybrid sliding-window/global-attention cache support. No new GGML operations or Metal kernels were added.

The initial release is based on llama.cpp commit `41abbfd599fbdd3470fcae0a1fb6530ad8403cd7`. Validation primarily covers Apple M2 Max with Metal, plus short CPU checks. Other backends have not been validated for this port.

## Quick start on Apple Silicon

Build this fork with CMake and an installed C/C++ toolchain. The download command uses the Hugging Face CLI (`hf`).

```bash
git clone --branch feature/zgcm https://github.com/ghthbl/llama-zgcm.git
cd llama-zgcm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --config Release -j 8 --target llama-server llama-cli llama-quantize llama-imatrix

hf download 0xtb/ZGCM-1-7B-GGUF ZGCM-1-7B-Q6_K-IMATRIX-BF16-down28-31.gguf --local-dir .

./build/bin/llama-server \
    -m ZGCM-1-7B-Q6_K-IMATRIX-BF16-down28-31.gguf \
    -c 131072 -ngl 99 -fa on -np 1 --jinja \
    --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> for the built-in chat interface. The local OpenAI-compatible API is at `http://127.0.0.1:8080/v1`.

Set context explicitly. The example uses 131,072 tokens; use `-c 262144` for the model's full context when memory permits. Use the embedded chat template. The [tagged release](https://github.com/ghthbl/llama-zgcm/releases/tag/zgcm-v0.1.0) and model card identify the exact source commit used for the published artifacts.

## Read the limitations before use

- **Repeated `!` output:** the model can degenerate into exclamation marks. This has also been observed with the original BF16 model. The model card documents our measurements and links an independent report from the NVFP4 conversion.
- **Quantization needs BF16 protection:** retain FFN down-projections 28-31 in BF16. Use the explicit tensor override in the model card when making your own quants.
- **Fidelity is not task accuracy:** matching reference predictions does not guarantee useful answers. Exploratory SVG generation was poor even with the reference and without Hermes. Reliable agent tool calling has not been established.

See the [model card](https://huggingface.co/0xtb/ZGCM-1-7B-GGUF) for the evidence, tested settings and limits of the evaluation.

## Please use it, improve it, and take it upstream

If you understand this implementation and find it useful, you are encouraged to use it, modify it, redistribute it, or adapt it for a contribution to the official llama.cpp project. No separate permission from this fork's maintainer is needed, and this fork imposes no additional restrictions beyond the applicable MIT licenses.

Please retain the applicable copyright and license notices. Anyone taking the work upstream should review and test it, take responsibility for the code they submit, and follow [upstream's contribution guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md). Upstream maintainers decide whether and how to accept support. Reuse and contributions are welcome; they are not a condition of using this fork.

## Credits and license

- **llama.cpp and GGML:** the original ggml authors and contributors. Their [MIT license](LICENSE) and copyright notice are retained.
- **ZGCM-1-7B:** ZGCAGI and the original model contributors. The model's separate MIT notice is included with the [GGUF distribution](https://huggingface.co/0xtb/ZGCM-1-7B-GGUF/blob/main/LICENSE).
- **Community conversion, port and evaluation:** [0xtb](https://huggingface.co/0xtb) / [ghthbl](https://github.com/ghthbl), with substantial assistance from Claude and OpenAI assistants under the publisher's direction. The publisher is responsible for this release.

## Original llama.cpp documentation

The upstream project supports many other models and platforms. Its **[official README](https://github.com/ggml-org/llama.cpp#readme)**, **[documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs)** and **[releases](https://github.com/ggml-org/llama.cpp/releases)** belong to the upstream project. Refer to the ZGCM instructions above for this fork.
