use polars::prelude::{DataFrame, Float32Type, Float64Type};

use LgbmError;
use {LabelVec, Matrixf64};

pub(crate) fn dataframe_to_mat(
    dataframe: &mut DataFrame,
    label_column: &str,
) -> Result<(Matrixf64, LabelVec), LgbmError> {
    let label_col_name = label_column;
    let (m, n) = dataframe.shape();
    let label_series = &dataframe.select_series(label_col_name)?[0].cast::<Float32Type>()?;
    if label_series.null_count() != 0 {
        return Err(LgbmError::new("Cannot create a dataset with null values, encountered nulls when creating the label array"));
    }

    dataframe.drop_in_place(label_col_name)?;

    let mut label_values = Vec::with_capacity(m);

    let label_values_ca = label_series.unpack::<Float32Type>()?;

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
            return Err(LgbmError::new("Cannot create a dataset with null values, encountered nulls when creating the label array"));
        }

        let series = series.cast::<Float64Type>()?;
        let ca = series.unpack::<Float64Type>()?;

        ca.into_no_null_iter()
            .enumerate()
            .for_each(|(row_idx, val)| feature_values[row_idx].push(val));
    }
    Ok((feature_values, label_values))
}

#[cfg(test)]
mod tests {
    use dataset::dataframe::dataframe_to_mat;
    use polars::df;
    use polars::prelude::*;

    #[test]
    fn simple_df() {
        let mut df: DataFrame = df![
            "feature_1" => [1.0, 0.7, 0.9, 0.2, 0.1],
            "feature_2" => [0.1, 0.4, 0.8, 0.2, 0.7],
            "feature_3" => [0.2, 0.5, 0.5, 0.1, 0.1],
            "feature_4" => [0.1, 0.1, 0.1, 0.7, 0.9],
           "label" => [0.0, 0.0, 0.0, 1.0, 1.0]
        ]
        .unwrap();
        let label_column = "label";
        let (x, y) = dataframe_to_mat(&mut df, label_column).unwrap();

        let recovered_feature: Vec<f64> = df
            .column("feature_1")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .map(|o| o.unwrap())
            .collect();

        assert_eq!(recovered_feature[0], x[0][0]);
    }
}
