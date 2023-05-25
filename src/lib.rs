extern crate libc;
extern crate lightgbm_sys;
extern crate serde_json;

type InputMatrix = Vec<Vec<f64>>;
type OutputVec = Vec<f32>;

extern crate alloc;
#[cfg(feature = "dataframe")]
extern crate polars;

macro_rules! lgbm_call {
    ($x:expr) => {
        LgbmError::check_return_value(unsafe { $x })
    };
}

mod error;
pub use error::{LgbmError, Result};

mod booster;
mod dataset;
