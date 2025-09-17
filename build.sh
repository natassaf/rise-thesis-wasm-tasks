#!/bin/bash
set -e  # Exit on any error

echo "Building inference-tasks..."
cd image_classification
cargo component build --release
cp target/wasm32-wasip1/release/inference_component_onnx.wasm ../../rise-thesis/wasm-modules/
echo "inference-tasks build done."


echo "Building matrix multiplication component..."
cd ../matrix-multiplication-component
cargo component build --target wasm32-wasip1 --release
cp wit/matrix.wit ../../rise-thesis/wasm-modules/
cp target/wasm32-wasip1/release/matrix_multiplication_component.wasm ../../rise-thesis/wasm-modules/
echo "math-tasks-components build done."


echo "All builds completed successfully."
