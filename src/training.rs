use crate::attention::softmax;
use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};

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

pub(crate) fn run(profile: &Profile) {
    let vocab = &profile.vocab;
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

    for epoch in 0..100 {
        let print = epoch % 10 == 0;
        if print {
            println!("epoch={}", epoch);
        }

        for &(input_id, target_id) in examples {
            let embedding_vec = embeddings.row(input_id);
            let mut score = output_weights.matvec(embedding_vec);
            let mut prob = softmax(&score);
            let mut loss = -prob[target_id].ln();
            let mut selected_token_id = argmax(&prob);
            if print {
                println!(
                    "input={} target={} selected={} prob={:?} loss={}",
                    vocab[input_id], vocab[target_id], vocab[selected_token_id], prob, loss
                );
            }

            let learning_rate = 0.1;

            for i in 0..output_weights.rows {
                let target = if i == target_id { 1.0 } else { 0.0 };
                let error = prob[i] - target;
                let row = output_weights.row_mut(i);

                for j in 0..row.len() {
                    row[j] -= learning_rate * error * embedding_vec[j];
                }
            }

            score = output_weights.matvec(embedding_vec);
            prob = softmax(&score);
            loss = -prob[target_id].ln();
            selected_token_id = argmax(&prob);

            if print {
                println!(
                    "input={} target={} selected={} prob={:?} loss={}",
                    vocab[input_id], vocab[target_id], vocab[selected_token_id], prob, loss
                );
            }
        }
    }

    for i in 0..output_weights.rows {
        println!("{}: {:?}", vocab[i], output_weights.row(i));
    }
}
