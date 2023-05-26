use serde_json::Value;

use booster::Booster;
use dataset::DataSet;
use Matrixf64;
use {booster, LgbmError};

/////////////////////////////////////////////
// types for training set
#[derive(Clone)]
pub struct TrainDataAdded(DataSet); // this should not implement default, so it can safely be used for construction
#[derive(Default, Clone)]
pub struct TrainDataMissing;
/////////////////////////////////////////////

/////////////////////////////////////////////
// types for params
#[derive(Clone)]
pub struct ParamsAdded(String, i32); // this should not implement default, so it can safely be used for construction
#[derive(Default, Clone)]
pub struct ParamsMissing;
/////////////////////////////////////////////

/// Builder for the Booster.
///
/// Uses TypeState Pattern to make sure that Training Data is added
/// so that Validation can be synced properly and params are present for training.
#[derive(Default, Clone)]
pub struct BoosterBuilder<T: Clone, P: Clone> {
    train_data: T,
    val_data: Vec<DataSet>,
    params: P,
}

/// These Methods are always available to the Booster.
impl<T: Clone, P: Clone> BoosterBuilder<T, P> {
    /// Returns the Builder and a clone from it. Useful if you want to train 2 models with
    /// only a couple differences. This should be called at the end of the adapter chain,
    /// where u defined all things that are equal in the models.
    /// U can then continue to build the models separately.
    ///
    /// ```
    /// use lightgbm::booster::Booster;
    /// use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// let params_a = serde_json::json! {
    ///             {
    ///                 "num_iterations": 5,
    ///                 "objective": "binary",
    ///                 "metric": "auc",
    ///                 "data_random_seed": 0
    ///             }
    ///         };
    /// let params_b = serde_json::json! {
    ///             {
    ///                 "num_iterations": 100,
    ///                 "objective": "binary",
    ///                 "metric": "acc",
    ///                 "data_random_seed": 42
    ///             }
    ///         };
    /// let x = vec![
    ///             vec![1.0, 0.1, 0.2, 0.1],
    ///             vec![0.7, 0.4, 0.5, 0.1],
    ///             vec![0.9, 0.8, 0.5, 0.1],
    ///             vec![0.2, 0.2, 0.8, 0.7],
    ///             vec![0.1, 0.7, 1.0, 0.9]];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let train_data = DataSet::from_mat(x,y);
    /// let (booster_low_it, booster_high_it) = Booster::builder()
    ///     .add_train_data(train_data)
    ///     .duplicate();
    /// let booster_low_it = booster_low_it
    ///     .add_params(params_a)?
    ///     .fit()?;
    /// let booster_high_it = booster_high_it
    ///     .add_params(params_b)?
    ///     .fit()?;
    /// # Ok(())}
    /// ```
    pub fn duplicate(self) -> (Self, Self) {
        (self.clone(), self)
    }
}

/// Methods in this block require, that no params are added to the Booster.
impl<T: Clone> BoosterBuilder<T, ParamsMissing> {
    /// Adds params to the Booster.
    /// Returns Error, if param parsing returns Error.
    ///
    /// ```
    /// use lightgbm::booster::Booster;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// let params = serde_json::json! {
    ///             {
    ///                 "num_iterations": 5,
    ///                 "objective": "binary",
    ///                 "metric": "auc",
    ///                 "data_random_seed": 0
    ///             }
    ///         };
    /// let booster_builder = Booster::builder().add_params(params)?;
    /// # Ok(())}
    /// ```
    pub fn add_params(self, params: Value) -> Result<BoosterBuilder<T, ParamsAdded>, LgbmError> {
        let num_iterations = params
            .get("num_iterations")
            .ok_or(LgbmError::new("Num iterations in params missing."))?
            .as_i64()
            .ok_or(LgbmError::new("Invalid Value for num iterations."))?;
        let parsed_params = parse_params(params)?;
        Ok(BoosterBuilder {
            params: ParamsAdded(parsed_params, num_iterations as i32),
            train_data: self.train_data,
            val_data: self.val_data,
        })
    }
}

/// Methods in this Block require, that there is no train data added to the Booster.
impl<P: Clone> BoosterBuilder<TrainDataMissing, P> {
    /// Adds training data. necessary for validation data (so bins can be synced)
    /// and for model fitting.
    /// ```
    /// use lightgbm::booster::Booster;
    /// use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// let params = serde_json::json! {
    ///             {
    ///                 "num_iterations": 5,
    ///                 "objective": "binary",
    ///                 "metric": "auc",
    ///                 "data_random_seed": 0
    ///             }
    ///         };
    /// let x = vec![
    ///             vec![1.0, 0.1, 0.2, 0.1],
    ///             vec![0.7, 0.4, 0.5, 0.1],
    ///             vec![0.9, 0.8, 0.5, 0.1],
    ///             vec![0.2, 0.2, 0.8, 0.7],
    ///             vec![0.1, 0.7, 1.0, 0.9]];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let train_data = DataSet::from_mat(x,y);
    /// let booster = Booster::builder()
    ///     .add_train_data(train_data)
    ///     .add_params(params)?
    ///     .fit()?;
    ///    
    /// # Ok(())}
    /// ```
    pub fn add_train_data(self, train: DataSet) -> BoosterBuilder<TrainDataAdded, P> {
        BoosterBuilder {
            train_data: TrainDataAdded(train),
            val_data: self.val_data,
            params: self.params,
        }
    }
}

/// Methods in this impl Block require, that training data is already added.
impl<P: Clone> BoosterBuilder<TrainDataAdded, P> {
    /// Adds validation data to the Booster.
    /// ```
    /// use lightgbm::booster::Booster;
    /// use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// let params = serde_json::json! {
    ///             {
    ///                 "num_iterations": 5,
    ///                 "objective": "binary",
    ///                 "metric": "auc",
    ///                 "data_random_seed": 0
    ///             }
    ///         };
    /// let x = vec![
    ///             vec![1.0, 0.1, 0.2, 0.1],
    ///             vec![0.7, 0.4, 0.5, 0.1],
    ///             vec![0.9, 0.8, 0.5, 0.1],
    ///             vec![0.2, 0.2, 0.8, 0.7],
    ///             vec![0.1, 0.7, 1.0, 0.9]];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let train_data = DataSet::from_mat(x,y);
    /// let x = vec![
    ///             vec![8.0, 0.2, 0.4, 0.5],
    ///             vec![0.9, 0.4, 0.3, 0.5],
    ///             vec![0.5, 0.6, 0.3, 0.8],
    ///             vec![0.244, 0.25, 0.9, 0.9],
    ///             vec![0.4, 0.8, 0.8, 0.7],
    ///         ];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let validation_data = DataSet::from_mat(x,y);
    /// let booster = Booster::builder()
    ///     .add_train_data(train_data)    // add training data first
    ///     .add_val_data(validation_data) // then validation data
    ///     .add_params(params)?
    ///     .fit()?;
    ///    
    /// # Ok(())}
    /// ```
    pub fn add_val_data(mut self, val: DataSet) -> Self {
        self.val_data.push(val);
        self
    }
}

/// Methods in this impl block are only available, after Training Data and Params are added.
impl BoosterBuilder<TrainDataAdded, ParamsAdded> {
    /// Builds the booster by:
    /// 1. Adding the training data
    /// 2. Adding the validation data
    /// 3. Training with the params
    ///
    /// Each of these steps can fail and return errors.
    ///
    /// ```
    /// use lightgbm::booster::Booster;
    /// use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// let params = serde_json::json! {
    ///             {
    ///                 "num_iterations": 5,
    ///                 "objective": "binary",
    ///                 "metric": "auc",
    ///                 "data_random_seed": 0
    ///             }
    ///         };
    /// let x = vec![
    ///             vec![1.0, 0.1, 0.2, 0.1],
    ///             vec![0.7, 0.4, 0.5, 0.1],
    ///             vec![0.9, 0.8, 0.5, 0.1],
    ///             vec![0.2, 0.2, 0.8, 0.7],
    ///             vec![0.1, 0.7, 1.0, 0.9]];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let train_data = DataSet::from_mat(x,y);
    /// let x = vec![
    ///             vec![8.0, 0.2, 0.4, 0.5],
    ///             vec![0.9, 0.4, 0.3, 0.5],
    ///             vec![0.5, 0.6, 0.3, 0.8],
    ///             vec![0.244, 0.25, 0.9, 0.9],
    ///             vec![0.4, 0.8, 0.8, 0.7],
    ///         ];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let validation_data = DataSet::from_mat(x,y);
    /// let booster = Booster::builder()
    ///     .add_train_data(train_data)     // this is necessary
    ///     .add_val_data(validation_data)  // this is optional
    ///     .add_params(params)?            // this is also necessary
    ///     .fit()?;
    ///    
    /// # Ok(())}
    /// ```
    pub fn fit(self) -> Result<Booster, LgbmError> {
        let train_data = self.train_data.0.load(None)?;
        let booster_handle = booster::ffi::new_booster(train_data.handle, &self.params.0)?;
        let mut validation_sets = Vec::with_capacity(self.val_data.len());
        for val in self.val_data.into_iter() {
            let loaded_data = val.load(Some(train_data.handle))?;
            booster::ffi::add_validation_data_to_booster(booster_handle, loaded_data.handle)?;
            validation_sets.push(loaded_data);
        }
        let mut booster = Booster {
            handle: booster_handle,
            train_data: Some(train_data),
            validation_data: validation_sets,
        };
        booster.train_loop(self.params.1)?; // param parsing checked already if present
        Ok(booster)
    }

    /// Build the Booster with fit and immediately predict for the given input.
    /// Can Fail in fit if the Booster isn't correctly build or in predict if the Input Data
    /// is corrupted.
    ///
    /// ```
    /// use lightgbm::booster::Booster;
    /// use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// let params = serde_json::json! {
    ///             {
    ///                 "num_iterations": 5,
    ///                 "objective": "binary",
    ///                 "metric": "auc",
    ///                 "data_random_seed": 0
    ///             }
    ///         };
    /// let x = vec![
    ///             vec![1.0, 0.1, 0.2, 0.1],
    ///             vec![0.7, 0.4, 0.5, 0.1],
    ///             vec![0.9, 0.8, 0.5, 0.1],
    ///             vec![0.2, 0.2, 0.8, 0.7],
    ///             vec![0.1, 0.7, 1.0, 0.9]];
    /// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let train_data = DataSet::from_mat(x,y);
    /// let input = vec![
    ///             vec![8.0, 0.2, 0.4, 0.5],
    ///             vec![0.9, 0.4, 0.3, 0.5],
    ///             vec![0.5, 0.6, 0.3, 0.8],
    ///             vec![0.244, 0.25, 0.9, 0.9],
    ///             vec![0.4, 0.8, 0.8, 0.7],
    ///         ];
    /// let (booster, pred) = Booster::builder()
    ///     .add_train_data(train_data)     
    ///     .add_params(params)?
    ///     .fit_predict(&input)?;
    ///
    /// assert_eq!(input.len(), pred[0].len()); // binary classification. One output value for each input vec
    /// # Ok(())}
    /// ```
    pub fn fit_predict(self, x: &Matrixf64) -> Result<(Booster, Matrixf64), LgbmError> {
        let booster = self.fit()?;
        let y = booster.predict(x)?;
        Ok((booster, y))
    }
}

/// Transforms a serde_json Value object into a String that Lightgbm Requires. Note that a conversion
/// to a CString is still required for the ffi.
/// The algorithms transforms data like this:
/// {"x": "y", "z": 1} => "x="y" z=1"
/// and
/// {"k" = ["a", "b"]} => "k="a,b""
/// Returns Error if the Value object somehow doesn't represents valid json, or the num_iterations
/// param is not set.
fn parse_params(params: Value) -> Result<String, LgbmError> {
    if params.get("num_iterations").is_none() {
        return Err(LgbmError::new("Num Iterations not specified."));
    }

    let s = params
        .as_object()
        .ok_or(LgbmError::new("Couldn't parse params"))?
        .iter()
        .map(|(k, v)| match v {
            Value::Array(a) => {
                let v_formatted = a.iter().map(|x| x.to_string() + ",").collect::<String>();
                let v_formatted = v_formatted
                    .replace("\",\"", ",")
                    .trim_end_matches(',')
                    .to_string();
                (k, v_formatted)
            }
            _ => (k, v.to_string()),
        })
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(" ");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use booster::builder::parse_params;
    use booster::Booster;
    use dataset::DataSet;
    use serde_json::json;
    use {LabelVec, Matrixf64};

    fn get_simple_params() -> serde_json::Value {
        json! {
            {
                "num_iterations": 5,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        }
    }

    fn get_dummy_data_1() -> (Matrixf64, LabelVec) {
        let data = vec![
            vec![1.0, 0.1, 0.2, 0.1],
            vec![0.7, 0.4, 0.5, 0.1],
            vec![0.9, 0.8, 0.5, 0.1],
            vec![0.2, 0.2, 0.8, 0.7],
            vec![0.1, 0.7, 1.0, 0.9],
        ];
        let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        (data, label)
    }

    fn get_dummy_data_2() -> (Matrixf64, LabelVec) {
        let data = vec![
            vec![8.0, 0.2, 0.4, 0.5],
            vec![0.9, 0.4, 0.3, 0.5],
            vec![0.5, 0.6, 0.3, 0.8],
            vec![0.244, 0.25, 0.9, 0.9],
            vec![0.4, 0.8, 0.8, 0.7],
        ];
        let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        (data, label)
    }

    #[test]
    fn simple_build_test() {
        let (train_x, train_y) = get_dummy_data_1();
        let (val_x, val_y) = get_dummy_data_2();
        let params = get_simple_params();

        let train_set = DataSet::from_mat(train_x, train_y);
        let val_set = DataSet::from_mat(val_x, val_y);

        let booster = Booster::builder()
            .add_train_data(train_set)
            .add_val_data(val_set)
            .add_params(params)
            .unwrap()
            .fit()
            .unwrap();

        let result_train = booster.get_eval_result_for_dataset(0).unwrap();
        let result_val = booster.get_eval_result_for_dataset(1).unwrap();
        assert!(booster.get_eval_result_for_dataset(2).is_err());

        assert_eq!(result_train.len(), 1);
        assert_eq!(result_val.len(), 1);
    }

    #[test]
    fn more_params() {
        let params = json! {
            {
                "num_iterations": 30,
                "objective": "binary",
                "boosting_type": "gbdt",
                "metrics": ["binary_logloss","auc"],
                "label_column": 0,
                "max_bin": 255,
                "tree_learner": "serial",
                "feature_fraction": 0.8,
                "is_enable_sparse": true,
                "data_random_seed": 0
            }
        };
        let (train_x, train_y) = get_dummy_data_1();
        let (val_x, val_y) = get_dummy_data_2();

        let train_set = DataSet::from_mat(train_x, train_y);
        let val_set = DataSet::from_mat(val_x, val_y);
        let val_set_2 = val_set.clone();

        let booster = Booster::builder()
            .add_train_data(train_set)
            .add_val_data(val_set)
            .add_val_data(val_set_2)
            .add_params(params)
            .unwrap()
            .fit()
            .unwrap();

        let result_train = booster.get_eval_result_for_dataset(0).unwrap();
        let result_val_1 = booster.get_eval_result_for_dataset(1).unwrap();
        let result_val_2 = booster.get_eval_result_for_dataset(2).unwrap();
        assert!(booster.get_eval_result_for_dataset(3).is_err());

        assert_eq!(result_train.len(), 2);
        assert_eq!(result_val_1.len(), 2);
        assert_eq!(result_val_2.len(), 2);
        let delta = (result_val_1[0].score - result_val_2[0].score).abs(); // floating point error
        assert!(-0.1 < delta && delta < 0.01);
    }

    #[test]
    fn params_test_valid() {
        let params = json! {
            {
                "num_iterations": 30,
                "objective": "binary",
                "metrics": ["binary_logloss","auc"],
                "is_enable_sparse": true
            }
        };
        let supposed_to_be =
            "is_enable_sparse=true metrics=\"binary_logloss,auc\" num_iterations=30 objective=\"binary\"";
        let parsed = parse_params(params).unwrap();

        assert_eq!(&parsed, supposed_to_be);
    }

    #[test]
    fn params_num_it_missing() {
        let params = json! {
            {
                "objective": "binary",
                "metrics": ["binary_logloss","auc"],
                "is_enable_sparse": true
            }
        };
        assert!(parse_params(params).is_err());
    }
}
