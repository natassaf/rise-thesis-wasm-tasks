

wit_bindgen::generate!({
    world: "matrix",
    path: "wit/matrix.wit",
});

struct Component;

// We use the types defined in the `wit` file.
// The `wit_bindgen::generate!` macro creates the `Guest` trait for us,
// with the `run` function signature matching our WIT.
impl Guest for Component {
    fn run(input: WasmEvent) -> WasmResult {
        // The `wasm-event` record has a single field: `event: string`.
        // We'll assume the matrices are passed as a JSON string within the `event` field.
        // You'll need to parse this string to get your matrices.
        let event_data = &input.event;

        // For this example, we'll assume the matrices are encoded in a specific JSON format.
        // E.g., `{"mat1":[[1,2],[3,4]], "mat2":[[5,6],[7,8]]}`
        let parsed_data: serde_json::Value = serde_json::from_str(event_data).unwrap();
        let mat1: Vec<Vec<f32>> = serde_json::from_value(parsed_data["mat1"].clone()).unwrap();
        let mat2: Vec<Vec<f32>> = serde_json::from_value(parsed_data["mat2"].clone()).unwrap();

        let mat1_rows = mat1.len();
        let mat1_cols = mat1[0].len();
        let mat2_cols = mat2[0].len();

        let mut result = vec![vec![0.0f32; mat2_cols]; mat1_rows];

        for i in 0..mat1_rows {
            for j in 0..mat2_cols {
                for k in 0..mat1_cols {
                    result[i][j] += mat1[i][k] * mat2[k][j];
                }
            }
        }

        // Convert the result back into a string to fit the `wasm-result` record.
        let output_string = serde_json::to_string(&result).unwrap();

        // Return a `WasmResult` record with the calculated output.
        WasmResult { output: output_string }
    }
}

export!(Component);


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiplication() {
        // 1. Define the input matrices as Rust vectors.
        let mat1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mat2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        // 2. Construct the JSON string to pass as the input event.
        let input_json = json!({
            "mat1": mat1,
            "mat2": mat2
        }).to_string();

        // 3. Create the WasmEvent instance with the JSON string.
        let input_event = WasmEvent {
            event: input_json,
        };

        // 4. Call the `run` function via the `Component` struct.
        // The `wit_bindgen` macro places the `run` function on the `Guest` trait,
        // which is implemented for the `Component` struct.
        let output_result = Component::run(input_event);

        // 5. Parse the output string from the WasmResult into a Rust vector.
        let actual_result: Vec<Vec<f32>> = serde_json::from_str(&output_result.output).unwrap();

        // 6. Define the expected result of the multiplication.
        // [[(1*5)+(2*7)], [(1*6)+(2*8)]] = [[19, 22]]
        // [[(3*5)+(4*7)], [(3*6)+(4*8)]] = [[43, 50]]
        let expected_result = vec![vec![19.0, 22.0], vec![43.0, 50.0]];
        println!("Expected result: {:?}", expected_result);
        // 7. Assert that the actual output matches the expected output.
        assert_eq!(actual_result, expected_result);
    }
}