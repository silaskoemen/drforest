use drforest_core::split_candidate_positions as core_split_candidate_positions;
use drforest_core::{
    SplitError,
    best_adaptive_complex_embedding_split as core_best_adaptive_complex_embedding_split,
    best_cart_split as core_best_cart_split,
    best_cart_split_one_feature as core_best_cart_split_one_feature,
    best_complex_embedding_split as core_best_complex_embedding_split,
    best_sliced_wasserstein_split as core_best_sliced_wasserstein_split,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn map_split_error(error: SplitError) -> PyErr {
    let message = match error {
        SplitError::InvalidDim => "dim must be positive",
        SplitError::InvalidMinLeaf => "min_leaf must be >= 1",
        SplitError::InvalidMaxCutpoints => "max_cutpoints must be >= 1 or None",
        SplitError::InvalidFeature => "feature index out of range",
        SplitError::ShapeMismatch => "x length, y length, and dim are inconsistent",
    };
    PyValueError::new_err(message)
}

fn contiguous_error(name: &str) -> PyErr {
    PyValueError::new_err(format!("{name} must be C-contiguous"))
}

#[pyfunction]
#[pyo3(signature = (xs, min_leaf, lo, hi, max_cutpoints = None))]
fn split_candidate_positions(
    xs: PyReadonlyArray1<'_, f64>,
    min_leaf: usize,
    lo: f64,
    hi: f64,
    max_cutpoints: Option<usize>,
) -> PyResult<Vec<usize>> {
    let xs = xs.as_slice().map_err(|_| contiguous_error("xs"))?;
    core_split_candidate_positions(xs, min_leaf, lo, hi, max_cutpoints).map_err(map_split_error)
}

#[pyfunction]
#[pyo3(signature = (x, y, dim, min_leaf, lo, hi, max_cutpoints = None))]
fn best_cart_split_one_feature(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    dim: usize,
    min_leaf: usize,
    lo: f64,
    hi: f64,
    max_cutpoints: Option<usize>,
) -> PyResult<Option<(f64, f64)>> {
    let x = x.as_slice().map_err(|_| contiguous_error("x"))?;
    let y_shape = y.shape();
    if y_shape.len() != 2 || y_shape[1] != dim {
        return Err(PyValueError::new_err("y shape and dim are inconsistent"));
    }
    let y = y.as_slice().map_err(|_| contiguous_error("y"))?;
    let split = core_best_cart_split_one_feature(x, y, dim, min_leaf, lo, hi, max_cutpoints)
        .map_err(map_split_error)?;
    Ok(split.map(|value| (value.threshold, value.score)))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x,
    y,
    features,
    min_leaf,
    bounds = None,
    max_cutpoints = None
))]
fn best_cart_split(
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    features: Vec<usize>,
    min_leaf: usize,
    bounds: Option<PyReadonlyArray2<'_, f64>>,
    max_cutpoints: Option<usize>,
) -> PyResult<Option<(usize, f64, f64)>> {
    let x_shape = x.shape();
    let y_shape = y.shape();
    if x_shape.len() != 2 || y_shape.len() != 2 || x_shape[0] != y_shape[0] {
        return Err(PyValueError::new_err("x and y shapes are inconsistent"));
    }
    let n_rows = x_shape[0];
    let n_cols = x_shape[1];
    let dim = y_shape[1];
    let bounds_slice = match bounds.as_ref() {
        Some(values) => {
            let shape = values.shape();
            if shape.len() != 2 || shape[0] != features.len() || shape[1] != 2 {
                return Err(PyValueError::new_err(
                    "bounds must have shape (len(features), 2)",
                ));
            }
            Some(values.as_slice().map_err(|_| contiguous_error("bounds"))?)
        }
        None => None,
    };
    let x = x.as_slice().map_err(|_| contiguous_error("x"))?;
    let y = y.as_slice().map_err(|_| contiguous_error("y"))?;
    let split = core_best_cart_split(
        x,
        n_rows,
        n_cols,
        y,
        dim,
        &features,
        min_leaf,
        bounds_slice,
        max_cutpoints,
    )
    .map_err(map_split_error)?;
    Ok(split.map(|value| (value.feature, value.threshold, value.score)))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x,
    psi_re,
    psi_im,
    features,
    scale,
    min_leaf,
    bounds = None,
    max_cutpoints = None
))]
fn best_complex_embedding_split(
    x: PyReadonlyArray2<'_, f64>,
    psi_re: PyReadonlyArray2<'_, f64>,
    psi_im: PyReadonlyArray2<'_, f64>,
    features: Vec<usize>,
    scale: f64,
    min_leaf: usize,
    bounds: Option<PyReadonlyArray2<'_, f64>>,
    max_cutpoints: Option<usize>,
) -> PyResult<Option<(usize, f64, f64)>> {
    let x_shape = x.shape();
    let re_shape = psi_re.shape();
    let im_shape = psi_im.shape();
    if x_shape.len() != 2
        || re_shape.len() != 2
        || im_shape.len() != 2
        || x_shape[0] != re_shape[0]
        || re_shape != im_shape
    {
        return Err(PyValueError::new_err(
            "x, psi_re, and psi_im shapes are inconsistent",
        ));
    }
    let n_rows = x_shape[0];
    let n_cols = x_shape[1];
    let n_embed = re_shape[1];
    let bounds_slice = match bounds.as_ref() {
        Some(values) => {
            let shape = values.shape();
            if shape.len() != 2 || shape[0] != features.len() || shape[1] != 2 {
                return Err(PyValueError::new_err(
                    "bounds must have shape (len(features), 2)",
                ));
            }
            Some(values.as_slice().map_err(|_| contiguous_error("bounds"))?)
        }
        None => None,
    };
    let x = x.as_slice().map_err(|_| contiguous_error("x"))?;
    let psi_re = psi_re.as_slice().map_err(|_| contiguous_error("psi_re"))?;
    let psi_im = psi_im.as_slice().map_err(|_| contiguous_error("psi_im"))?;
    let split = core_best_complex_embedding_split(
        x,
        n_rows,
        n_cols,
        psi_re,
        psi_im,
        n_embed,
        &features,
        scale,
        min_leaf,
        bounds_slice,
        max_cutpoints,
    )
    .map_err(map_split_error)?;
    Ok(split.map(|value| (value.feature, value.threshold, value.score)))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x,
    psi_re,
    psi_im,
    selected_features,
    features,
    min_leaf,
    bounds = None,
    max_cutpoints = None
))]
fn best_adaptive_complex_embedding_split(
    x: PyReadonlyArray2<'_, f64>,
    psi_re: PyReadonlyArray2<'_, f64>,
    psi_im: PyReadonlyArray2<'_, f64>,
    selected_features: usize,
    features: Vec<usize>,
    min_leaf: usize,
    bounds: Option<PyReadonlyArray2<'_, f64>>,
    max_cutpoints: Option<usize>,
) -> PyResult<Option<(usize, f64, f64)>> {
    let x_shape = x.shape();
    let re_shape = psi_re.shape();
    let im_shape = psi_im.shape();
    if x_shape.len() != 2
        || re_shape.len() != 2
        || im_shape.len() != 2
        || x_shape[0] != re_shape[0]
        || re_shape != im_shape
    {
        return Err(PyValueError::new_err(
            "x, psi_re, and psi_im shapes are inconsistent",
        ));
    }
    let n_rows = x_shape[0];
    let n_cols = x_shape[1];
    let n_embed = re_shape[1];
    let bounds_slice = match bounds.as_ref() {
        Some(values) => {
            let shape = values.shape();
            if shape.len() != 2 || shape[0] != features.len() || shape[1] != 2 {
                return Err(PyValueError::new_err(
                    "bounds must have shape (len(features), 2)",
                ));
            }
            Some(values.as_slice().map_err(|_| contiguous_error("bounds"))?)
        }
        None => None,
    };
    let x = x.as_slice().map_err(|_| contiguous_error("x"))?;
    let psi_re = psi_re.as_slice().map_err(|_| contiguous_error("psi_re"))?;
    let psi_im = psi_im.as_slice().map_err(|_| contiguous_error("psi_im"))?;
    let split = core_best_adaptive_complex_embedding_split(
        x,
        n_rows,
        n_cols,
        psi_re,
        psi_im,
        n_embed,
        selected_features,
        &features,
        min_leaf,
        bounds_slice,
        max_cutpoints,
    )
    .map_err(map_split_error)?;
    Ok(split.map(|value| (value.feature, value.threshold, value.score)))
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x,
    projected,
    features,
    min_leaf,
    bounds = None,
    max_cutpoints = None
))]
fn best_sliced_wasserstein_split(
    x: PyReadonlyArray2<'_, f64>,
    projected: PyReadonlyArray2<'_, f64>,
    features: Vec<usize>,
    min_leaf: usize,
    bounds: Option<PyReadonlyArray2<'_, f64>>,
    max_cutpoints: Option<usize>,
) -> PyResult<Option<(usize, f64, f64)>> {
    let x_shape = x.shape();
    let projected_shape = projected.shape();
    if x_shape.len() != 2 || projected_shape.len() != 2 || x_shape[0] != projected_shape[0] {
        return Err(PyValueError::new_err(
            "x and projected shapes are inconsistent",
        ));
    }
    let n_rows = x_shape[0];
    let n_cols = x_shape[1];
    let n_projections = projected_shape[1];
    let bounds_slice = match bounds.as_ref() {
        Some(values) => {
            let shape = values.shape();
            if shape.len() != 2 || shape[0] != features.len() || shape[1] != 2 {
                return Err(PyValueError::new_err(
                    "bounds must have shape (len(features), 2)",
                ));
            }
            Some(values.as_slice().map_err(|_| contiguous_error("bounds"))?)
        }
        None => None,
    };
    let x = x.as_slice().map_err(|_| contiguous_error("x"))?;
    let projected = projected
        .as_slice()
        .map_err(|_| contiguous_error("projected"))?;
    let split = core_best_sliced_wasserstein_split(
        x,
        n_rows,
        n_cols,
        projected,
        n_projections,
        &features,
        min_leaf,
        bounds_slice,
        max_cutpoints,
    )
    .map_err(map_split_error)?;
    Ok(split.map(|value| (value.feature, value.threshold, value.score)))
}

#[pymodule]
fn _drforest_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(split_candidate_positions, module)?)?;
    module.add_function(wrap_pyfunction!(best_cart_split_one_feature, module)?)?;
    module.add_function(wrap_pyfunction!(best_cart_split, module)?)?;
    module.add_function(wrap_pyfunction!(best_complex_embedding_split, module)?)?;
    module.add_function(wrap_pyfunction!(
        best_adaptive_complex_embedding_split,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(best_sliced_wasserstein_split, module)?)?;
    Ok(())
}
