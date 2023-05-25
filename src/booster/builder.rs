use serde_json::Value;

use booster::Booster;
use dataset::DataSet;
use {booster, LgbmError};
use {LabelVec, Matrixf64};

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
    pub fn duplicate(self) -> (Self, Self) {
        (self.clone(), self)
    }
}

/// Methods in this block require, that no params are added to the Booster.
impl<T: Clone> BoosterBuilder<T, ParamsMissing> {
    /// Adds params to the Booster.
    /// Returns Error, if param parsing returns Error.
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
    pub fn fit_predict(self, x: &Matrixf64) -> Result<(Booster, Matrixf64), LgbmError> {
        let booster = self.fit()?;
        let y = booster.predict(x)?;
        Ok((booster, y))
    }
}

/// Transforms a serde_json Value object into a String that Lightgbm Requires. Note that a conversion
/// to a CString is still required for the ffi.
/// The algorithms thransforms data like this:
/// {"x": "y", "z": 1} => "x=y z=1"
/// and
/// {"k" = ["a", "b"]} => "k=a,b"
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
    use serde_json::{json, Value};

    use booster::Booster;
    use dataset::{DataFormat, DataSet};

    fn _default_params() -> Value {
        let params = json! {
            {
                "num_iterations": 1,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        params
    }

    #[test]
    fn easy() {
        let x = vec![vec![1.0, 1.0, 0.5], vec![1.0, 1.0, 0.5]];
        let y = vec![0_f32, 1.0];
        let format = DataFormat::Vecs { x, y };
        let dataset = DataSet::new(format);
        let (bst_low_lr, bst_high_lr) = Booster::builder().add_train_data(dataset).duplicate();
        let _bst_low_lr = bst_low_lr
            .add_params(_default_params())
            .unwrap()
            .fit()
            .unwrap();
        let _bst_high_lr = bst_high_lr
            .add_params(_default_params())
            .unwrap()
            .fit()
            .unwrap();
    }
}
