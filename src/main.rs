fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0;

    for i in 0..a.len() {
	sum += a[i] * b[i];
    }

    sum
}


struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize
}

impl Matrix {
    fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
	assert_eq!(data.len(), (rows * cols));
	Self { data, rows, cols }
    }
    
    fn matvec(&self, x: &[f32]) -> Vec<f32> {
	assert!(self.cols > 0);
	assert_eq!(self.data.len() % self.cols, 0);
	assert_eq!(self.cols, x.len());

	let mut result = Vec::new();

	for i in 0..self.rows {
	    let row = self.row(i);
	    result.push(dot(row, x));
	}

	result
    }

    fn row(&self, index: usize) -> &[f32] {
	assert!(index < self.rows);
	let start = index*self.cols;
	&self.data[start..start+self.cols]
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
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

fn matvec(matrix: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();

    for i in 0..matrix.len() {
	let row = &matrix[i];
	result.push(dot(row, x))
    }

    result
}

fn dispvec(vec: &[f32]) {
    for v in vec {
	println!("{}", v);
    }
}

fn main() {
    println!("{}", dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]));

    let embeddings = Matrix::new(
	3,
	2,
	vec![
	    0.1, 0.2, // token id 0
	    0.3, 0.4, // token id 1
	    0.5, 0.6, // token id 2
	],
    );

    let token_id = 1;
    let x = embeddings.row(token_id);
    dispvec(x);

    let weights = Matrix::new(
	2, 2,
	vec![
	    1.0, 2.0,
	    3.0, 4.0
	]
    );

    let y = weights.matvec(x);
    dispvec(&y);

    let output_weights = Matrix::new(
	3, 2,
	vec![
	    1.0, 0.0, // calculate score id 0
	    0.0, 1.0, // id 1
	    1.0, 1.0, // id 2
	]
    );

    let score = output_weights.matvec(embeddings.row(1));
    let prob = softmax(&score);
    dispvec(&prob);

    let selected_token_id = argmax(&prob);
    println!("{}", selected_token_id);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matvec_multiplies_each_row() {
	let matrix = Matrix::new(
	    2,
	    2,
	    vec![1.0, 2.0, 3.0, 4.0]
	);
	let result = matrix.matvec(&[1.0, 2.0]);
	assert_eq!(result, vec![5.0, 11.0]);
    }

    #[test]
    fn matvec_rectanguler() {
	let matrix = Matrix::new(
	    2,
	    3,
	    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	);
	let result = matrix.matvec(&[1.0, 2.0, 3.0]);
	assert_eq!(result, vec![14.0, 32.0]);
    }

    #[test]
    #[should_panic]
    fn matrix_rejects_wrong_element_count() {
	Matrix::new(2, 3, vec![1.0, 2.0]);
    }

    #[test]
    fn row_returns_requested_row() {
	let matrix = Matrix::new(
	    2, 3,
	    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	);

	assert_eq!(matrix.row(0), &[1.0, 2.0, 3.0]);
	assert_eq!(matrix.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn softmax_test() {
	assert_eq!(softmax(&Vec::from(&[0.0, 0.0])), &[0.5, 0.5]);
	assert_eq!(softmax(&Vec::from(&[-1000.0, -1000.0])), &[0.5, 0.5]);

	let result = softmax(&Vec::from(&[2.0, 1.0, 0.0]));
	let expected = [0.665241, 0.244728, 0.090031];

	for i in 0..expected.len() {
	    assert!((result[i] - expected[i]).abs() < 0.00001);
	}
    }
}
