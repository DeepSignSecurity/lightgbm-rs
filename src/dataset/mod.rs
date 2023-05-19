#[cfg(feature = "dataframe")]
mod dataframe;
mod ffi;

use lightgbm_sys::DatasetHandle;
#[cfg(feature = "dataframe")]
use polars::prelude::DataFrame;

use OutputVec;
use {Error, InputMatrix};

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
        x: InputMatrix,
        y: OutputVec,
    },
    #[cfg(feature = "dataframe")]
    DataFrame {
        df: DataFrame,
        y_column: String,
    },
}

pub struct LoadedDataSet {
    pub(crate) handle: DatasetHandle,
}

impl Drop for LoadedDataSet {
    fn drop(&mut self) {
        ffi::drop_dataset(self.handle).expect("Something went wrong dropping the Dataset.");
    }
}

impl DataSet {
    pub(crate) fn load(&self, reference: &Option<DatasetHandle>) -> Result<LoadedDataSet, Error> {
        let handle = match &self.format {
            DataFormat::File { path } => ffi::load_dataset_from_file(path, &self.params, reference),
            DataFormat::Vecs { x, y } => ffi::load_from_vec(x, y, &self.params, reference),
            #[cfg(feature = "dataframe")]
            DataFormat::DataFrame { df, y_column } => {
                let (x, y) = dataframe::dataframe_to_mat(dataframe, label_column)?;
                ffi::load_from_vec(&x, &y, &self.params, reference)
            }
        }?;
        Ok(LoadedDataSet { handle })
    }
}
