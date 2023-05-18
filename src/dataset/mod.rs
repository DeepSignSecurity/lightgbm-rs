mod ffi;

use lightgbm_sys::DatasetHandle;
#[cfg(feature = "dataframe")]
use polars::prelude::*;

/// Represents an unloaded Dataset for the Booster Builder.
/// At the fit step of the BoosterBuilder, these will be added to the
/// lightgbm backend
#[derive(Clone)]
pub struct DataSet {
    format: DataFormat,
    /// Possible params for the Dataset. This is !currently! not in use
    /// and will be just an empty string
    params: String,
}

#[derive(Clone)]
pub enum DataFormat {
    File {
        path: String,
    },
    Vecs {
        x: Vec<Vec<f64>>,
        y: Vec<f32>,
    },
    #[cfg(feature = "dataframe")]
    DataFrame {
        df: DataFrame,
        y_column: Into<String>,
    },
}

pub struct LoadedDataSet {
    pub(crate) handle: DatasetHandle,
}

impl Drop for LoadedDataSet {
    fn drop(&mut self) {
        todo!()
    }
}

impl DataSet {
    pub(crate) fn load(&self, reference: Option<DatasetHandle>) -> LoadedDataSet {
        match &self.format {
            DataFormat::File { path } => todo!(), //add here corresponding ffi calls
            DataFormat::Vecs { x, y } => todo!(),
            #[cfg(feature = "dataframe")]
            DataFormat::DataFrame { df, y_column } => todo!(),
        }
    }
}
