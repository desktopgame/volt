mod matrix;
use matrix::Matrix;

mod attention;
use attention::{attend_tokens, softmax};

fn argmax(x: &[f32]) -> usize {
    assert!(!x.is_empty());

    let mut best = 0;

    for i in 0..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }

    best
}

fn predict_next(
    token_ids: &[usize],
    embeddings: &Matrix,
    w_query: &Matrix,
    w_key: &Matrix,
    w_value: &Matrix,
    output_weights: &Matrix,
) -> usize {
    let x = attend_tokens(token_ids, embeddings, w_query, w_key, w_value);
    let score = output_weights.matvec(&x);
    let prob = softmax(&score);
    let next_id = argmax(&prob);

    println!("input: {:?}, attention: {:?}", token_ids, x);
    next_id
}

fn run_generation() {
    let embeddings = Matrix::new(
        3,
        2,
        vec![
            0.1, 0.2, // token id 0
            0.3, 0.4, // token id 1
            0.5, 0.6, // token id 2
        ],
    );

    let output_weights = Matrix::new(
        3,
        2,
        vec![
            1.0, 0.0, // calculate score id 0
            0.0, 1.0, // id 1
            1.0, 1.0, // id 2
        ],
    );

    let mut token_ids = vec![1];
    let w_query = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let w_key = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let w_value = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    for _ in 0..5 {
        let next_id = predict_next(
            &token_ids,
            &embeddings,
            &w_query,
            &w_key,
            &w_value,
            &output_weights,
        );
        token_ids.push(next_id);
    }
    println!("{:?}", token_ids);
}

mod training;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(String::as_str) {
        None => run_generation(),
        Some("train") => training::run(),
        Some(_) => eprintln!("help: cargo run -- [train]"),
    }
}
