//! Functionality related to errors and error handling.

use std::error;
use std::ffi::{CStr, NulError};
use std::fmt::{self, Debug, Display};

use lightgbm_sys;

#[cfg(feature = "dataframe")]
use polars::prelude::*;

/// Convenience return type for most operations which can return an `LightGBM`.
pub type Result<T> = std::result::Result<T, LgbmError>;

/// Wrap errors returned by the LightGBM library.
#[derive(Debug, Eq, PartialEq)]
pub struct LgbmError {
    desc: String,
}

impl LgbmError {
    pub(crate) fn new<S: Into<String>>(desc: S) -> Self {
        Self { desc: desc.into() }
    }

    /// Check the return value from an LightGBM FFI call, and return the last error message on error.
    ///
    /// Return values of 0 are treated as success, returns values of -1 are treated as errors.
    ///
    /// Meaning of any other return values are undefined, and will cause a panic.
    pub(crate) fn check_return_value(ret_val: i32) -> Result<()> {
        match ret_val {
            0 => Ok(()),
            -1 => Err(Self::from_lightgbm()),
            _ => panic!("unexpected return value '{}', expected 0 or -1", ret_val),
        }
    }

    /// Get the last error message from LightGBM.
    fn from_lightgbm() -> Self {
        let c_str = unsafe { CStr::from_ptr(lightgbm_sys::LGBM_GetLastError()) };
        let str_slice = c_str.to_str().unwrap();
        Self::new(str_slice)
    }
}

impl error::Error for LgbmError {}

impl Display for LgbmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LightGBM error: {}", &self.desc)
    }
}

impl From<NulError> for LgbmError {
    fn from(_: NulError) -> Self {
        Self {
            desc: "Null Byte found within String".into(),
        }
    }
}

#[cfg(feature = "dataframe")]
impl From<PolarsError> for LgbmError {
    fn from(pe: PolarsError) -> Self {
        Self {
            desc: pe.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_value_handling() {
        let result = LgbmError::check_return_value(0);
        assert_eq!(result, Ok(()));

        let result = LgbmError::check_return_value(-1);
        assert_eq!(result, Err(LgbmError::new("Everything is fine")));
    }
}
