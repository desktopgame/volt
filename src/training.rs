use crate::attention::softmax;
use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Profile {
    vocab: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    examples: Vec<(usize, usize)>,
}

// TODO: copy from main.rs
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

pub(crate) fn run(profile: &Profile, verbose: Option<PathBuf>) -> std::io::Result<()> {
    let rows = profile.embeddings.len();
    let cols = profile
        .embeddings
        .first()
        .expect("embeddings is empty")
        .len();
    assert!(cols > 0);
    assert!(profile.embeddings.iter().all(|row| row.len() == cols));
    let flatten_embeddings = profile.embeddings.iter().flatten().copied().collect();
    let embeddings = Matrix::new(rows, cols, flatten_embeddings);

    let vocab_size = profile.vocab.len();
    assert_eq!(rows, vocab_size);
    let mut output_weights = Matrix::new(vocab_size, cols, vec![0.0; vocab_size * cols]);
    let examples = &profile.examples;

    let mut verbose_file = verbose.map(File::create).transpose()?;
    if let Some(vf) = verbose_file.as_mut() {
        writeln!(vf, "epoch,row,col,weight")?;
    }

    println!("epoch,input_id,target_id,target_prob,loss");
    for epoch in 0..100 {
        for &(input_id, target_id) in examples {
            let embedding_vec = embeddings.row(input_id);
            let score = output_weights.matvec(embedding_vec);
            let prob = softmax(&score);
            let learning_rate = 0.1;

            for i in 0..output_weights.rows {
                let target = if i == target_id { 1.0 } else { 0.0 };
                let error = prob[i] - target;
                let row = output_weights.row_mut(i);

                for j in 0..row.len() {
                    row[j] -= learning_rate * error * embedding_vec[j];
                }
            }
        }
        for &(input_id, target_id) in examples {
            let x = embeddings.row(input_id);
            let scores = output_weights.matvec(x);
            let prob = softmax(&scores);
            let loss = -prob[target_id].ln();

            println!(
                "{},{},{},{},{}",
                epoch, input_id, target_id, prob[target_id], loss
            );
        }

        if let Some(vf) = verbose_file.as_mut() {
            for i in 0..output_weights.rows {
                for j in 0..output_weights.cols {
                    writeln!(vf, "{},{},{},{}", epoch, i, j, output_weights.row(i)[j])?;
                }
            }
        }
    }
    Ok(())
}
