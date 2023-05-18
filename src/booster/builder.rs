use serde_json::Value;

use booster::Booster;
use dataset::DataSet;
use Error;
use {InputMatrix, OutputVec};

// types for training set
#[derive(Clone)]
pub struct TrainDataAdded(DataSet); // this should not implement default, so it can safely be used for construction
#[derive(Default, Clone)]
pub struct TrainDataNotAdded;

// types for params
#[derive(Clone)]
pub struct ParamsAdded(Value); // this should not implement default, so it can safely be used for construction
#[derive(Default, Clone)]
pub struct ParamsNotAdded;

/// Builder for the Booster.
///
/// Uses TypeState Pattern to make sure that Training Data is added
/// so that Validation can be synced properly and params are present for training.
#[derive(Default, Clone)]
pub struct BoosterBuilder<T: Clone, P: Clone> {
    train_data: T,
    val_data: Vec<DataSet>,
    params: P, // after #3 should this be a struct
}

impl<T: Clone, P: Clone> BoosterBuilder<T, P> {
    /// Returns the Builder and a clone from it. Useful if you want to train 2 models with
    /// only a couple differences
    pub fn duplicate(self) -> (Self, Self) {
        (self.clone(), self)
    }
}

impl<T: Clone> BoosterBuilder<T, ParamsNotAdded> {
    pub fn add_params(self, params: Value) -> BoosterBuilder<T, ParamsAdded> {
        BoosterBuilder {
            params: ParamsAdded(params),
            train_data: self.train_data,
            val_data: self.val_data,
        }
    }
}

impl<P: Clone> BoosterBuilder<TrainDataNotAdded, P> {
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

impl<P: Clone> BoosterBuilder<TrainDataAdded, P> {
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
    /// Each of these steps can return errors.
    pub fn fit(self) -> Result<Booster, Error> {
        let train = self.train_data.0.load(None);
        let vals: Vec<_> = self
            .val_data
            .iter()
            .map(|v| v.load(Some(train.handle)))
            .collect();
        // train classifier
        // call train ffi from here

        // return
        todo!()
    }

    pub fn fit_predict(self, x: &InputMatrix) -> Result<(Booster, OutputVec), Error> {
        let booster = self.fit()?;
        let y = booster.predict(x)?;
        Ok((booster, y))
    }
}
