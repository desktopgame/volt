fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0;

    for i in 0..a.len() {
	sum += a[i] * b[i];
    }

    sum
}

fn matvec(matrix: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    let mut result = Vec::new();

    for i in 0..matrix.len() {
	let row = &matrix[i];
	result.push(dot(row, x))
    }

    result
}

fn matvec_flat(matrix: &[f32], cols: usize, x: &[f32]) -> Vec<f32> {
    assert!(cols > 0);
    assert_eq!(matrix.len() % cols, 0);
    assert_eq!(cols, x.len());

    let mut result = Vec::new();

    for i in (0..matrix.len()).step_by(cols) {
	let row = &matrix[i..i+cols];
	result.push(dot(row, x));
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

    let result = matvec_flat(&[1.0, 2.0, 3.0, 4.0], 2, &[1.0, 2.0]);
    dispvec(&result);
    dispvec(&result);
    println!("Hello, world!");
}
