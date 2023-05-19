use polars::prelude::{DataFrame, Float32Type, Float64Type, PolarsError};

use {InputMatrix, OutputVec};

type FfiError = crate::Error;

pub(crate) fn dataframe_to_mat(
    &mut dataframe: DataFrame,
    label_column: String,
) -> Result<(InputMatrix, OutputVec), FfiError> {
    let label_col_name = label_column.as_str();
    let (m, n) = dataframe.shape();
    let label_series = &dataframe.select_series(label_col_name)?[0].cast::<Float32Type>()?;
    if label_series.null_count() != 0 {
        return Err(FfiError::new("Cannot create a dataset with null values, encountered nulls when creating the label array"));
    }

    dataframe
        .drop_in_place(label_col_name)
        .map_err(FfiError::new)?;

    let mut label_values = Vec::with_capacity(m);

    let label_values_ca = label_series
        .unpack::<Float32Type>()
        .map_err(FfiError::new)?;

    label_values_ca
        .into_no_null_iter()
        .enumerate()
        .for_each(|(_row_idx, val)| {
            label_values.push(val);
        });

    let mut feature_values = Vec::with_capacity(m);
    for _i in 0..m {
        feature_values.push(Vec::with_capacity(n));
    }

    for (_col_idx, series) in dataframe.get_columns().iter().enumerate() {
        if series.null_count() != 0 {
            return Err(FfiError::new("Cannot create a dataset with null values, encountered nulls when creating the label array"));
        }

        let series = series.cast::<Float64Type>().map_err(FfiError::new)?;
        let ca = series.unpack::<Float64Type>().map_err(FfiError::new)?;

        ca.into_no_null_iter()
            .enumerate()
            .for_each(|(row_idx, val)| feature_values[row_idx].push(val));
    }
    Ok((feature_values, label_values))
}
