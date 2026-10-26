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

	for i in (0..self.data.len()).step_by(self.cols) {
	    let row = &self.data[i..i+self.cols];
	    result.push(dot(row, x));
	}

	result
    }
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

    let matrix = Matrix::new(
	2,
	2,
	vec![1.0, 2.0, 3.0, 4.0],
    );
    let result = matrix.matvec(&[1.0, 2.0]);
    dispvec(&result);
    println!("Hello, world!");
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
}
