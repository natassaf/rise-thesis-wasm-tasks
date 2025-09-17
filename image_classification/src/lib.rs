#![allow(unused_braces)]
use image::ImageReader;
use image::{DynamicImage, RgbImage};
use ndarray::{Array, Dim};
use std::fs;
use std::io::{BufRead, Cursor};
use base64::{Engine as _, engine::general_purpose};

fn decode_base64_image(base64_string: &str) -> Result<Vec<u8>, base64::DecodeError> {
    // Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    let base64_data = if base64_string.contains(",") {
        base64_string.split(",").nth(1).unwrap_or(base64_string)
    } else {
        base64_string
    };
    
    // Decode the base64 string back to binary data
    let decoded_bytes = general_purpose::STANDARD.decode(base64_data)?;
    Ok(decoded_bytes)
}

wit_bindgen::generate!({
    path: "wit",
    world: "ml",
});

use self::wasi::nn::{
    graph::{Graph, load, ExecutionTarget, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};


// Implement the exported run function
struct Component;

impl Guest for Component {
    
    fn run(event: WasmEvent) -> WasmResult {
        let event_data = &event.event;
        let parsed_data: serde_json::Value = serde_json::from_str(event_data).unwrap();
        let model_path: String = serde_json::from_value(parsed_data["model_path"].clone()).unwrap();
        let labels_path: String = serde_json::from_value(parsed_data["labels_path"].clone()).unwrap();
        let image_input = decode_base64_image(&parsed_data["input"].as_str().unwrap()).unwrap();

        let model_bytes = fs::read(&model_path).expect("Failed to read model");
        println!("Loaded model bytes: {}", model_bytes.len());

        // Load the graph
        let graph = load(&[model_bytes], GraphEncoding::Onnx, ExecutionTarget::Cpu)
            .expect("Failed to load graph");
       
        println!("Graph loaded into WASI-NN");

        // Load labels (assuming labels are in same directory as model with .txt extension)
        let labels_file = fs::read(&labels_path).expect("Failed to read labels");
        let labels: Vec<String> = labels_file.lines()
            .map(|l| l.unwrap())
            .collect();
        println!("Loaded {} labels", labels.len());
        let event_data = &event.event;
        let parsed_data: serde_json::Value = serde_json::from_str(event_data).unwrap();
        // let img_path: String = serde_json::from_value(parsed_data["image_path"].clone()).unwrap();

        let num_labels = 1000;
 
        let exec_context = Graph::init_execution_context(&graph).unwrap();

        // Prepare tensor
        let dimensions: TensorDimensions = vec![1, 3, 224, 224];
        let data: TensorData = image_to_tensor(image_input, 224, 224);
        let tensor = Tensor::new(&dimensions, TensorType::Fp32, &data);
        let input_tensor: Vec<(String, Tensor)> = vec![("data".to_string(), tensor)];

        // Compute
        let output_tensor_vec = exec_context.compute(input_tensor).unwrap();
        let output_tensor = output_tensor_vec.iter().find_map(|(name, tensor)| {
            if name == "squeezenet0_flatten0_reshape0" { Some(tensor) } else { None }
        }).expect("No output tensor found");

        let output_f32 = bytes_to_f32_vec(output_tensor.data().to_vec());

        // Reshape output to (1,1000,1,1)
        let output_tensor = Array::from_shape_vec([1, num_labels, 1, 1], output_f32).unwrap();

        // Softmax
        let exp_output = output_tensor.mapv(|x| x.exp());
        let sum_exp_output = exp_output.sum_axis(ndarray::Axis(1));
        let softmax_output = exp_output / &sum_exp_output;

        // Top-3 predictions
        let mut sorted = softmax_output
            .axis_iter(ndarray::Axis(1))
            .enumerate()
            .map(|(i, v)| (i, v[Dim([0, 0, 0])]))
            .collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top3 = sorted.iter().take(3)
            .map(|(index, prob)| format!("{}: {:.5}", labels[*index], prob))
            .collect::<Vec<_>>()
            .join(", ");

        drop(exec_context);
        drop(graph);
        WasmResult { output: top3 }

    }


    
}


pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    data.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn image_to_tensor(image_bytes: Vec<u8>, height: u32, width: u32) -> Vec<u8> {
    let cursor = Cursor::new(image_bytes);
    let pixels = ImageReader::new(cursor).with_guessed_format().unwrap().decode().unwrap();
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


export!(Component);