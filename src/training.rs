use crate::attention::softmax;
use crate::matrix::Matrix;

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


pub(crate) fn run() {
    let vocab = ["猫", "犬", "鳴く", "吠える"];
    let embeddings = Matrix::new(
	4, 2,
	vec![
	    1.0, 0.0, // 猫
	    0.0, 1.0, // 犬
	    0.0, 0.0, // 鳴く
	    0.0, 0.0, // 吠える
	]
    );

    let mut output_weights = Matrix::new(4, 2, vec![0.0; 8]);
    let examples: [(usize, usize); 2] = [
	(0, 2), // 猫→鳴く
	(1, 3), // 犬→吠える
    ];

    for epoch in 0..100 {
	let print = epoch % 10 == 0;
	if print {
	    println!("epoch={}", epoch);
	}

	for &(input_id, target_id) in &examples {
	    let embedding_vec = embeddings.row(input_id);
	    let mut score = output_weights.matvec(embedding_vec);
	    let mut prob = softmax(&score);
	    let mut selected_token_id = argmax(&prob);
	    if print {
		println!("input={} target={} selected={} prob={:?}", vocab[input_id], vocab[target_id], vocab[selected_token_id], prob);
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
	    selected_token_id = argmax(&prob);

	    if print {
		println!("input={} target={} selected={} prob={:?}", vocab[input_id], vocab[target_id], vocab[selected_token_id], prob);
	    }
	}
    }

    for i in 0..output_weights.rows {
	println!("{}: {:?}", vocab[i], output_weights.row(i));
    }
}
