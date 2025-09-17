To execute this we included a wasmtime executable on my-wasmtime folder that includes wasi-nn and ONNX support which are not prebuilt on the official wasmtime binaries. To get that we clone the wasmtime repo and run cargo build:

```bash
git clone https://github.com/bytecodealliance/wasmtime.git
cd wasmtime
cargo build --features component-model,wasi-nn,wasmtime-wasi-nn/onnx-download

To run the main of the project we do:

1. `cargo component build --release`
2. Run the executable with:

   ```bash
   my-wasmtime/wasmtime run \
     -Snn \
     --dir ./fixture::fixture \
     target/wasm32-wasip1/release/neural_network_wasm_task.wasm
