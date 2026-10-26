fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

pub(crate) struct Matrix {
    data: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Matrix {
    pub(crate) fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), (rows * cols));
        Self { data, rows, cols }
    }

    pub(crate) fn matvec(&self, x: &[f32]) -> Vec<f32> {
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

    pub(crate) fn row(&self, index: usize) -> &[f32] {
        assert!(index < self.rows);
        let start = index * self.cols;
        &self.data[start..start + self.cols]
    }

    pub(crate) fn row_mut(&mut self, index: usize) -> &mut [f32] {
        assert!(index < self.rows);
        let start = index * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matvec_multiplies_each_row() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = matrix.matvec(&[1.0, 2.0]);
        assert_eq!(result, vec![5.0, 11.0]);
    }

    #[test]
    fn matvec_rectanguler() {
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
        let matrix = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(matrix.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.row(1), &[4.0, 5.0, 6.0]);
    }
}
