# Voice Engine

A robust media processing library for Rust, designed for building voice applications.

## Features

- **Codecs**: Support for G.711 (PCMA/PCMU), G.722, G.729, and Opus.
- **Media Processing**: Includes Jitter Buffer, Resampling, and DTMF handling.
- **Voice Activity Detection (VAD)**: Integrated Silero VAD and Ten VAD. Both are pure Rust implementations without `onnxruntime` dependency, offering superior performance.
- **Noise Reduction**: Built-in denoiser.
- **Speech Services**:
  - **ASR**: Aliyun, Tencent.
  - **TTS**: Aliyun, Deepgram, Tencent.
- **Transport**: RTP and WebRTC support.

## Usage

This library is intended to be used as a component in voice applications like `rustpbx`.

## Performance

We have optimized the VAD engines to achieve state-of-the-art performance in pure Rust, significantly outperforming ONNX Runtime.

**Benchmark Environment**: macOS, Apple Silicon (Single Core), 60s 16kHz Audio

| VAD Engine     | Implementation       | Time (60s)   | FPS         | RTF (Real Time Factor) | Note                     |
| :------------- | :------------------- | :----------- | :---------- | :--------------------- | :----------------------- |
| **TinyTen**    | **Rust (Hand-opt)**  | **~31.5 ms** | **~59,600** | **~1900x**             | Extremely lightweight    |
| **TinySilero** | **Rust (Optimized)** | **~76.4 ms** | **~24,500** | **~785x**              | **>2x faster than ONNX** |
| ONNX Ten       | ONNX Runtime         | ~99.6 ms     | ~37,600     | ~760x                  |                          |
| ONNX Silero    | ONNX Runtime         | ~158.3 ms    | ~11,800     | ~380x                  | Standard baseline        |
| WebRTC VAD     | C/C++ (Bind)         | ~3.1 ms      | ~960,000    | ~20,000x               | Legacy, less accurate    |

> Note: The Rust implementations are fully specialized for the specific model architectures, using techniques like loop unrolling and unsafe pointer arithmetic to eliminate overhead.

