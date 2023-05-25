use booster::builder::{BoosterBuilder, ParamsMissing, TrainDataMissing};
use dataset::{DataSet, LoadedDataSet};
use LgbmError;
use {InputMatrix, OutputVec};

mod builder;
mod ffi;

/// Evaluation Result of a Booster on a given Dataset.
/// Returned by get_eval
pub struct EvalResult {
    metric_name: String,
    score: f64,
}

/// Class that is returned by the builder, once fit() is called.
/// Used to interact with a trained booster.
pub struct Booster {
    handle: lightgbm_sys::BoosterHandle,
    train_data: Option<LoadedDataSet>, // dont drop datasets
    validation_data: Vec<LoadedDataSet>,
}

// exchange params method as well? does this make sense?
impl Booster {
    /// Returns a builder. At least training data and params need to be added,
    /// so that the model can be fitted (built).
    pub fn builder() -> BoosterBuilder<TrainDataMissing, ParamsMissing> {
        BoosterBuilder::default()
    }

    /// Generates a prediction for a given Input.
    /// The Output has the same dimensions as the input,
    /// because this returns class probabilities.
    /// Can return an Error if the input or model is corrupt.
    pub fn predict(&self, x: &InputMatrix) -> Result<InputMatrix, LgbmError> {
        let prediction_params = ""; // do we need this?
        ffi::predict(self.handle, prediction_params, x)
    }

    /// Returns the scores for the train and validation set.
    /// If successful, returns a Result with a m·n matrix, where
    /// m = number of datasets
    /// n = number of metrics
    pub fn get_eval_results(&self) -> Result<Vec<Vec<EvalResult>>, LgbmError> {
        todo!("just ffi call i guess")
    }

    /// this should take &mut self, because it changes the model
    pub(crate) fn train_loop(&mut self, max_iterations: i32) -> Result<(), LgbmError> {
        let mut is_finished = 0;
        let mut i = 0;
        while is_finished == 0 && i < max_iterations {
            // callback stuff here
            ffi::train_one_step(self.handle, &mut is_finished)?;
            i += 1;
        }
        Ok(())
    }

    /// Train a booster further with a new dataset.
    /// This should not reset the already existing submodels.
    /// Pass an empty array as validation data, if you don't want to validate the train results.
    /// TODO validate this after implemented
    pub fn finetune(
        &mut self,
        _train_data: DataSet,
        _validation_data: Vec<DataSet>,
    ) -> Result<(), LgbmError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn simple() {}
}
