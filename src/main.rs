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

fn dispvec(vec: &[f32]) {
    for v in vec {
	println!("{}", v);
    }
}

fn main() {
    println!("{}", dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]));
    dispvec(&matvec(&[Vec::from([1.0, 2.0]), Vec::from([3.0, 4.0])], &[1.0, 2.0]));
    println!("Hello, world!");
}
