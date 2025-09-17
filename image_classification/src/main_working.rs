#![allow(unused_braces)]
use image::ImageReader;
use image::{DynamicImage, RgbImage};
use ndarray::{Array, Dim};
use std::fs;
use std::io::BufRead;

const IMG_PATH: &str = "fixture/images/dog.jpg";

wit_bindgen::generate!({
    path: "wit",
    world: "ml",
    generate_unused_types: true
});

use self::wasi::nn::{
    graph::{Graph, GraphBuilder, load, ExecutionTarget, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};


// Implement the exported run function
struct Ml;

impl Guest for Ml {
    fn run(input: WasmEvent) -> WasmResult {
        println!("Received event: {}", input.event);

        // Load the ONNX model - SqueezeNet 1.1-7
        let model: GraphBuilder = fs::read("fixture/models/squeezenet1.1-7.onnx").unwrap();
        println!("Read ONNX model, size in bytes: {}", model.len());

        let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Gpu).unwrap();
        println!("Loaded graph into wasi-nn");

        let exec_context = Graph::init_execution_context(&graph).unwrap();
        println!("Created wasi-nn execution context.");

        // Load SqueezeNet labels
        let labels = fs::read("fixture/labels/squeezenet1.1-7.txt").unwrap();
        let class_labels: Vec<String> = labels.lines().map(|line| line.unwrap()).collect();
        println!("Read ONNX Labels, # of labels: {}", class_labels.len());

        // Prepare WASI-NN tensor
        let dimensions: TensorDimensions = vec![1, 3, 224, 224];
        let data: TensorData = image_to_tensor(IMG_PATH.to_string(), 224, 224);
        let tensor = Tensor::new(&dimensions, TensorType::Fp32, &data);
        let input_tensor: Vec<(String, Tensor)> = vec![("data".to_string(), tensor)];

        // Execute inference
        let output_tensor_vec = exec_context.compute(input_tensor).unwrap();
        println!("Executed graph inference");

        let output_tensor = output_tensor_vec.iter().find_map(|(tensor_name, tensor)| {
            if tensor_name == "squeezenet0_flatten0_reshape0" {
                Some(tensor)
            } else {
                None
            }
        }).expect("No output tensor");
        let output_data = output_tensor.data();

        println!("Retrieved output data with length: {}", output_data.len());
        let output_f32 = bytes_to_f32_vec(output_data);

        let output_shape = [1, 1000, 1, 1];
        let output_tensor = Array::from_shape_vec(output_shape, output_f32).unwrap();

        // Compute softmax
        let exp_output = output_tensor.mapv(|x| x.exp());
        let sum_exp_output = exp_output.sum_axis(ndarray::Axis(1));
        let softmax_output = exp_output / &sum_exp_output;

        let mut sorted = softmax_output
            .axis_iter(ndarray::Axis(1))
            .enumerate()
            .map(|(i, v)| (i, v[Dim([0, 0, 0])]))
            .collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Format top-3 predictions
        let top3 = sorted.iter().take(3)
            .map(|(index, probability)| {
                format!("{}: {:.5}", class_labels[*index], probability)
            })
            .collect::<Vec<_>>()
            .join(", ");

        println!("Top 3 predictions: {}", top3);

        WasmResult { output: top3 }
    }
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    data.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = ImageReader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img: RgbImage = dyn_img.to_rgb8();

    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    let bytes_required = raw_u8_arr.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    for i in 0..raw_u8_arr.len() {
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let rgb_iter = i % 3;
        let norm_u8_f32: f32 = (u8_f32 / 255.0 - mean[rgb_iter]) / std[rgb_iter];
        let u8_bytes = norm_u8_f32.to_ne_bytes();

        for j in 0..4 {
            u8_f32_arr[(raw_u8_arr.len() * 4 * rgb_iter / 3) + (i / 3) * 4 + j] = u8_bytes[j];
        }
    }

    u8_f32_arr
}

pub fn main(){
    
}
export!(Ml);