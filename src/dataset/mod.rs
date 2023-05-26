use lightgbm_sys::DatasetHandle;
#[cfg(feature = "dataframe")]
use polars::prelude::DataFrame;

use LabelVec;
use {LgbmError, Matrixf64};

#[cfg(feature = "dataframe")]
mod dataframe;
mod ffi;

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

impl DataSet {
    // ignore params for now
    pub fn new(format: DataFormat) -> Self {
        Self {
            format,
            params: "".to_string(),
        }
    }

    /// Create a Dataset from a Matrix and label Vector.
    ///
    /// ```
    /// use lightgbm::dataset::DataSet;
    ///
    /// let x = vec![vec![0.3,0.4,0.8],vec![0.3,0.4,0.5]];
    /// let y = vec![0.0,1.0];
    /// let dataset = DataSet::from_mat(x, y);
    /// ```
    pub fn from_mat(x: Matrixf64, y: LabelVec) -> Self {
        let f = DataFormat::Vecs { x, y };
        Self::new(f)
    }

    /// Create a Dataset from a File Path.
    /// Does not check if the given Path is valid.
    /// This happens during Booster building.
    /// ```
    /// use lightgbm::dataset::DataSet;
    ///
    /// let dataset = DataSet::from_file("some/file.txt");
    /// ```
    pub fn from_file(path: impl Into<String>) -> Self {
        let f = DataFormat::File { path: path.into() };
        Self::new(f)
    }

    #[cfg(feature = "dataframe")]
    /// Creates a Dataset from a DataFrame.
    /// The label column corresponds to the column in the DataFrame,
    /// that contains the labels for the sample. The rest are features.
    /// ```
    /// extern crate polars;
    /// use lightgbm::dataset::DataSet;
    /// use polars::df;
    /// use polars::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let mut df: DataFrame = df![
    ///             "feature_1" => [1.0, 0.7, 0.9, 0.2, 0.1],
    ///             "feature_2" => [0.1, 0.4, 0.8, 0.2, 0.7],
    ///             "feature_3" => [0.2, 0.5, 0.5, 0.1, 0.1],
    ///             "feature_4" => [0.1, 0.1, 0.1, 0.7, 0.9],
    ///            "label" => [0.0, 0.0, 0.0, 1.0, 1.0]
    ///         ]?;
    ///     let label_column = "label";
    ///     let dataset = DataSet::from_data_frame(df, label_column);
    /// # Ok(())}
    pub fn from_data_frame(df: DataFrame, label_column: impl Into<String>) -> Self {
        let f = DataFormat::DataFrame {
            df,
            y_column: label_column.into(),
        };
        Self::new(f)
    }
}

/// Represents the different Formats for datasets, that can be loaded into lightgbm.
/// Depending on the type, a different way for processing/loading the data is chosen.
#[derive(Clone)]
pub enum DataFormat {
    File {
        path: String,
    },
    Vecs {
        x: Matrixf64,
        y: LabelVec,
    },
    #[cfg(feature = "dataframe")]
    DataFrame {
        df: DataFrame,
        y_column: String,
    },
}

/// Loaded Dataset. Created by calling the load method on a Dataset.
/// This is done by the BoosterBuilder.
/// The DatasetHandle is returned by the lightgbm ffi.
pub struct LoadedDataSet {
    pub(crate) handle: DatasetHandle,
    dataset: DataSet, // this can maybe be removed
}

/// Data needs to be freed manually
impl Drop for LoadedDataSet {
    fn drop(&mut self) {
        ffi::drop_dataset(self.handle).expect("Something went wrong dropping the Dataset.");
    }
}

impl DataSet {
    /// Load a Dataset into Lightgbm. Depending on the format, different ffis are used.
    /// Either returns a Loaded Dataset or an Error, if lightgbm (or polars) reject the data.
    /// This functions is called by the BoosterBuilder
    pub(crate) fn load(self, reference: Option<DatasetHandle>) -> Result<LoadedDataSet, LgbmError> {
        let handle = match &self.format {
            DataFormat::File { path } => ffi::load_dataset_from_file(path, &self.params, reference),
            DataFormat::Vecs { x, y } => ffi::load_from_vec(x, y, &self.params, reference),
            #[cfg(feature = "dataframe")]
            DataFormat::DataFrame { df, y_column } => {
                let (x, y) = dataframe::dataframe_to_mat(&mut df.clone(), y_column)?;
                ffi::load_from_vec(&x, &y, &self.params, reference)
            }
        }?;
        Ok(LoadedDataSet {
            handle,
            dataset: self,
        })
    }
}
