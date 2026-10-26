use crate::matrix::Matrix;

pub(crate) fn softmax(x: &[f32]) -> Vec<f32> {
    assert!(!x.is_empty());

    let mut max = x[0];
    let mut sum = 0.0;
    for &e in x {
	if e > max {
	    max = e;
	}
    }
    for e in x {
	sum += (e - max).exp();
    }

    let mut result = Vec::new();
    for e in x {
	result.push((e - max).exp() / sum);
    }

    result
}


fn weighted_sum(vectors: &Matrix, coefficients: &[f32]) -> Vec<f32> {
    assert_eq!(vectors.rows, coefficients.len());
    
    let mut results = vec![0.0; vectors.cols];

    for i in 0..vectors.rows {
	let row = vectors.row(i);

	for j in 0..vectors.cols {
	    results[j] += row[j] * coefficients[i];
	}
    }

    results
}

fn attention(
    query: &[f32],
    keys: &Matrix,
    values: &Matrix,
) -> Vec<f32> {
    assert!(!query.is_empty());
    assert_eq!(keys.cols, query.len());
    assert_eq!(keys.rows, values.rows);

    let mut scores = keys.matvec(query);
    for i in 0..scores.len() {
	let score = scores[i];
	scores[i] = score / (query.len() as f32).sqrt();
    }

    let coefficients = softmax(&scores);

    weighted_sum(values, &coefficients)
}

pub(crate) fn attend_tokens(
    token_ids: &[usize],
    embeddings: &Matrix,
    w_query: &Matrix,
    w_key: &Matrix,
    w_value: &Matrix
) -> Vec<f32> {
    assert!(!token_ids.is_empty());

    let mut key_data = Vec::new();
    let mut value_data = Vec::new();

    for &token_id in token_ids {
	let embedding_vec = embeddings.row(token_id);
	key_data.extend_from_slice(&w_key.matvec(embedding_vec));
	value_data.extend_from_slice(&w_value.matvec(embedding_vec));
    }

    let keys = Matrix::new(token_ids.len(), w_key.rows, key_data);
    let values = Matrix::new(token_ids.len(), w_value.rows, value_data);

    let last_token_id = token_ids.last();
    let query = w_query.matvec(embeddings.row(*last_token_id.unwrap()));

    attention(&query, &keys, &values)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn softmax_test() {
	assert_eq!(softmax(&[0.0, 0.0]), &[0.5, 0.5]);
	assert_eq!(softmax(&[-1000.0, -1000.0]), &[0.5, 0.5]);

	let result = softmax(&[2.0, 1.0, 0.0]);
	let expected = [0.665241, 0.244728, 0.090031];

	for i in 0..expected.len() {
	    assert!((result[i] - expected[i]).abs() < 0.00001);
	}
    }

    #[test]
    fn weighted_sum_combines_row() {
	let vectors = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
	let result = weighted_sum(&vectors, &[0.25, 0.75]);
	assert_eq!(result, vec![0.25, 1.5]);
    }

    #[test]
    fn attention_averages_values_for_equal_scores() {
	let keys = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
	let values = Matrix::new(2, 2, vec![2.0, 0.0, 0.0, 4.0]);

	let result = attention(&[1.0, 1.0], &keys, &values);

	assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn attend_tokens_combines_embeddings() {
	let embeddings = Matrix::new(2, 2, vec![2.0, 0.0, 0.0, 2.0]);
	let identity = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
	let result = attend_tokens(&[0, 1], &embeddings, &identity, &identity, &identity);

	assert!(result[1] > result[0]);
	assert!((result[0] + result[1] - 2.0).abs() < 0.0001);
    }
}
