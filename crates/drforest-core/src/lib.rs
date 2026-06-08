#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    pub threshold: f64,
    pub score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureSplit {
    pub feature: usize,
    pub threshold: f64,
    pub score: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitError {
    InvalidDim,
    InvalidMinLeaf,
    InvalidMaxCutpoints,
    InvalidFeature,
    ShapeMismatch,
}

pub fn best_cart_split_one_feature(
    x: &[f64],
    y: &[f64],
    dim: usize,
    min_leaf: usize,
    lo: f64,
    hi: f64,
    max_cutpoints: Option<usize>,
) -> Result<Option<Split>, SplitError> {
    if dim == 0 {
        return Err(SplitError::InvalidDim);
    }
    if min_leaf == 0 {
        return Err(SplitError::InvalidMinLeaf);
    }
    if matches!(max_cutpoints, Some(0)) {
        return Err(SplitError::InvalidMaxCutpoints);
    }
    if x.len() * dim != y.len() {
        return Err(SplitError::ShapeMismatch);
    }
    if x.len() < 2 * min_leaf {
        return Ok(None);
    }

    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));

    let xs: Vec<f64> = order.iter().map(|&idx| x[idx]).collect();
    let positions = split_candidate_positions(&xs, min_leaf, lo, hi, max_cutpoints)?;
    if positions.is_empty() {
        return Ok(None);
    }

    let mut total = vec![0.0; dim];
    for &row in &order {
        for output in 0..dim {
            total[output] += y[row * dim + output];
        }
    }

    let mut selected = vec![false; n - 1];
    for pos in positions {
        selected[pos] = true;
    }

    let mut prefix = vec![0.0; dim];
    let mut best: Option<Split> = None;
    for rank in 0..(n - 1) {
        let row = order[rank];
        for output in 0..dim {
            prefix[output] += y[row * dim + output];
        }
        if !selected[rank] {
            continue;
        }

        let n_left = rank + 1;
        let n_right = n - n_left;
        let mut sq_norm = 0.0;
        for output in 0..dim {
            let left_mean = prefix[output] / n_left as f64;
            let right_mean = (total[output] - prefix[output]) / n_right as f64;
            let diff = left_mean - right_mean;
            sq_norm += diff * diff;
        }
        let score = (n_left * n_right) as f64 / (n * n) as f64 * sq_norm;
        let lower = xs[rank].max(lo);
        let upper = xs[rank + 1].min(hi);
        let split = Split {
            threshold: 0.5 * (lower + upper),
            score,
        };
        if best.is_none_or(|current| split.score > current.score) {
            best = Some(split);
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
pub fn best_cart_split(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    dim: usize,
    features: &[usize],
    min_leaf: usize,
    bounds: Option<&[f64]>,
    max_cutpoints: Option<usize>,
) -> Result<Option<FeatureSplit>, SplitError> {
    if n_cols == 0 || dim == 0 {
        return Err(SplitError::InvalidDim);
    }
    if min_leaf == 0 {
        return Err(SplitError::InvalidMinLeaf);
    }
    if matches!(max_cutpoints, Some(0)) {
        return Err(SplitError::InvalidMaxCutpoints);
    }
    if x.len() != n_rows * n_cols || y.len() != n_rows * dim {
        return Err(SplitError::ShapeMismatch);
    }
    if bounds.is_some_and(|values| values.len() != 2 * features.len()) {
        return Err(SplitError::ShapeMismatch);
    }
    if n_rows < 2 * min_leaf {
        return Ok(None);
    }

    let mut best: Option<FeatureSplit> = None;
    let mut feature_x = vec![0.0; n_rows];
    for (feature_pos, &feature) in features.iter().enumerate() {
        if feature >= n_cols {
            return Err(SplitError::InvalidFeature);
        }
        for row in 0..n_rows {
            feature_x[row] = x[row * n_cols + feature];
        }
        let (lo, hi) = match bounds {
            Some(values) => (values[2 * feature_pos], values[2 * feature_pos + 1]),
            None => (f64::NEG_INFINITY, f64::INFINITY),
        };
        let Some(split) =
            best_cart_split_one_feature(&feature_x, y, dim, min_leaf, lo, hi, max_cutpoints)?
        else {
            continue;
        };
        let candidate = FeatureSplit {
            feature,
            threshold: split.threshold,
            score: split.score,
        };
        if best.is_none_or(|current| candidate.score > current.score) {
            best = Some(candidate);
        }
    }
    Ok(best)
}

pub fn split_candidate_positions(
    xs: &[f64],
    min_leaf: usize,
    lo: f64,
    hi: f64,
    max_cutpoints: Option<usize>,
) -> Result<Vec<usize>, SplitError> {
    let n = xs.len();
    if min_leaf == 0 {
        return Err(SplitError::InvalidMinLeaf);
    }
    if matches!(max_cutpoints, Some(0)) {
        return Err(SplitError::InvalidMaxCutpoints);
    }
    if n < 2 * min_leaf {
        return Ok(Vec::new());
    }

    let mut valid = Vec::new();
    for pos in 0..(n - 1) {
        let lower = xs[pos].max(lo);
        let upper = xs[pos + 1].min(hi);
        if pos >= min_leaf - 1 && pos < n - min_leaf && xs[pos] != xs[pos + 1] && lower < upper {
            valid.push(pos);
        }
    }

    let Some(max_cutpoints) = max_cutpoints else {
        return Ok(valid);
    };
    if valid.len() <= max_cutpoints {
        return Ok(valid);
    }

    let low = valid[0] as f64;
    let high = *valid.last().expect("valid is non-empty") as f64;
    let step = (high - low) / (2 * max_cutpoints) as f64;
    let mut snapped = Vec::with_capacity(max_cutpoints);
    for j in 0..max_cutpoints {
        let probe = low + (2 * j + 1) as f64 * step;
        let insertion = valid.partition_point(|&pos| (pos as f64) < probe);
        let chosen = if insertion == 0 {
            valid[0]
        } else if insertion == valid.len() {
            *valid.last().expect("valid is non-empty")
        } else {
            let left = valid[insertion - 1];
            let right = valid[insertion];
            if probe - left as f64 <= right as f64 - probe {
                left
            } else {
                right
            }
        };
        if snapped.last().is_none_or(|&last| last != chosen) {
            snapped.push(chosen);
        }
    }
    Ok(snapped)
}

#[allow(clippy::too_many_arguments)]
pub fn best_complex_embedding_split(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    psi_re: &[f64],
    psi_im: &[f64],
    n_embed: usize,
    features: &[usize],
    scale: f64,
    min_leaf: usize,
    bounds: Option<&[f64]>,
    max_cutpoints: Option<usize>,
) -> Result<Option<FeatureSplit>, SplitError> {
    if n_cols == 0 || n_embed == 0 {
        return Err(SplitError::InvalidDim);
    }
    if min_leaf == 0 {
        return Err(SplitError::InvalidMinLeaf);
    }
    if matches!(max_cutpoints, Some(0)) {
        return Err(SplitError::InvalidMaxCutpoints);
    }
    if x.len() != n_rows * n_cols
        || psi_re.len() != n_rows * n_embed
        || psi_im.len() != psi_re.len()
    {
        return Err(SplitError::ShapeMismatch);
    }
    if bounds.is_some_and(|values| values.len() != 2 * features.len()) {
        return Err(SplitError::ShapeMismatch);
    }
    if n_rows < 2 * min_leaf {
        return Ok(None);
    }

    let mut best: Option<FeatureSplit> = None;
    for (feature_pos, &feature) in features.iter().enumerate() {
        if feature >= n_cols {
            return Err(SplitError::InvalidFeature);
        }
        let (lo, hi) = match bounds {
            Some(values) => (values[2 * feature_pos], values[2 * feature_pos + 1]),
            None => (f64::NEG_INFINITY, f64::INFINITY),
        };
        let Some(split) = best_complex_embedding_split_one_feature(
            x,
            n_rows,
            n_cols,
            feature,
            psi_re,
            psi_im,
            n_embed,
            scale,
            min_leaf,
            lo,
            hi,
            max_cutpoints,
        )?
        else {
            continue;
        };
        let candidate = FeatureSplit {
            feature,
            threshold: split.threshold,
            score: split.score,
        };
        if best.is_none_or(|current| candidate.score > current.score) {
            best = Some(candidate);
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
pub fn best_complex_embedding_split_one_feature(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    feature: usize,
    psi_re: &[f64],
    psi_im: &[f64],
    n_embed: usize,
    scale: f64,
    min_leaf: usize,
    lo: f64,
    hi: f64,
    max_cutpoints: Option<usize>,
) -> Result<Option<Split>, SplitError> {
    if n_cols == 0 || n_embed == 0 {
        return Err(SplitError::InvalidDim);
    }
    if min_leaf == 0 {
        return Err(SplitError::InvalidMinLeaf);
    }
    if matches!(max_cutpoints, Some(0)) {
        return Err(SplitError::InvalidMaxCutpoints);
    }
    if feature >= n_cols {
        return Err(SplitError::InvalidFeature);
    }
    if x.len() != n_rows * n_cols
        || psi_re.len() != n_rows * n_embed
        || psi_im.len() != psi_re.len()
    {
        return Err(SplitError::ShapeMismatch);
    }
    if n_rows < 2 * min_leaf {
        return Ok(None);
    }

    let mut order: Vec<usize> = (0..n_rows).collect();
    order.sort_by(|&a, &b| {
        x[a * n_cols + feature]
            .total_cmp(&x[b * n_cols + feature])
            .then(a.cmp(&b))
    });

    let xs: Vec<f64> = order.iter().map(|&idx| x[idx * n_cols + feature]).collect();
    let positions = split_candidate_positions(&xs, min_leaf, lo, hi, max_cutpoints)?;
    if positions.is_empty() {
        return Ok(None);
    }

    let mut total_re = vec![0.0; n_embed];
    let mut total_im = vec![0.0; n_embed];
    for row in 0..n_rows {
        let offset = row * n_embed;
        for embed in 0..n_embed {
            total_re[embed] += psi_re[offset + embed];
            total_im[embed] += psi_im[offset + embed];
        }
    }

    let mut selected = vec![false; n_rows - 1];
    for pos in positions {
        selected[pos] = true;
    }

    let mut prefix_re = vec![0.0; n_embed];
    let mut prefix_im = vec![0.0; n_embed];
    let mut best: Option<Split> = None;
    for rank in 0..(n_rows - 1) {
        let row = order[rank];
        let offset = row * n_embed;
        for embed in 0..n_embed {
            prefix_re[embed] += psi_re[offset + embed];
            prefix_im[embed] += psi_im[offset + embed];
        }
        if !selected[rank] {
            continue;
        }

        let n_left = rank + 1;
        let n_right = n_rows - n_left;
        let mut sq_norm = 0.0;
        for embed in 0..n_embed {
            let left_re = prefix_re[embed] / n_left as f64;
            let left_im = prefix_im[embed] / n_left as f64;
            let right_re = (total_re[embed] - prefix_re[embed]) / n_right as f64;
            let right_im = (total_im[embed] - prefix_im[embed]) / n_right as f64;
            let diff_re = left_re - right_re;
            let diff_im = left_im - right_im;
            sq_norm += diff_re * diff_re + diff_im * diff_im;
        }
        let score = scale * (n_left * n_right) as f64 / (n_rows * n_rows) as f64 * sq_norm;
        let lower = xs[rank].max(lo);
        let upper = xs[rank + 1].min(hi);
        let split = Split {
            threshold: 0.5 * (lower + upper),
            score,
        };
        if best.is_none_or(|current| split.score > current.score) {
            best = Some(split);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_positions_match_python_contract_example() {
        let xs: Vec<f64> = (0..200).map(|value| value as f64).collect();
        let positions = split_candidate_positions(&xs, 1, 50.0, 120.0, Some(8)).unwrap();
        assert_eq!(positions, vec![54, 63, 72, 80, 89, 97, 106, 115]);
    }

    #[test]
    fn cart_split_finds_planted_feature_cut() {
        let x = [-2.0, -1.0, -0.5, 0.2, 0.7, 1.5];
        let y = [-1.0, -1.2, -0.8, 2.0, 2.2, 1.8];
        let split =
            best_cart_split_one_feature(&x, &y, 1, 2, f64::NEG_INFINITY, f64::INFINITY, None)
                .unwrap()
                .expect("split should exist");
        assert!(split.threshold > -0.5 && split.threshold < 0.2);
        assert!(split.score > 2.0);
    }

    #[test]
    fn bounded_split_uses_feasible_interval_midpoint() {
        let x = [0.0, 10.0];
        let y = [0.0, 1.0];
        let split = best_cart_split_one_feature(&x, &y, 1, 1, 9.0, 10.0, None)
            .unwrap()
            .expect("bounded split should exist");
        assert_eq!(split.threshold, 9.5);
    }

    #[test]
    fn invalid_min_leaf_is_an_error() {
        let xs = [0.0, 1.0];
        assert_eq!(
            split_candidate_positions(&xs, 0, f64::NEG_INFINITY, f64::INFINITY, None),
            Err(SplitError::InvalidMinLeaf)
        );
    }

    #[test]
    fn complex_embedding_split_matches_cart_with_zero_imaginary_part() {
        let x = [-2.0, -1.0, -0.5, 0.2, 0.7, 1.5];
        let y = [-1.0, -1.2, -0.8, 2.0, 2.2, 1.8];
        let zeros = vec![0.0; y.len()];

        let cart =
            best_cart_split_one_feature(&x, &y, 1, 2, f64::NEG_INFINITY, f64::INFINITY, None)
                .unwrap();
        let complex = best_complex_embedding_split_one_feature(
            &x,
            x.len(),
            1,
            0,
            &y,
            &zeros,
            1,
            1.0,
            2,
            f64::NEG_INFINITY,
            f64::INFINITY,
            None,
        )
        .unwrap();

        assert_eq!(complex, cart);
    }
}
