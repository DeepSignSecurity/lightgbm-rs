use booster::builder::{BoosterBuilder, ParamsMissing, TrainDataMissing};
use dataset::LoadedDataSet;
use LgbmError;
use Matrixf64;

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
    pub fn predict(&self, x: &Matrixf64) -> Result<Matrixf64, LgbmError> {
        let prediction_params = ""; // do we need this?
        ffi::predict(self.handle, prediction_params, x)
    }

    /// Returns the scores for a certain dataset. You can use the index like this:
    /// 0 = Train Dataset
    /// 1 = 1. Validation Dataset
    /// 2 = 2. Validation Dataset
    /// ...
    /// n = nth Validation Dataset
    pub fn get_eval_result_for_dataset(
        &self,
        dataset_index: i32,
    ) -> Result<Vec<EvalResult>, LgbmError> {
        if dataset_index > self.validation_data.len() as i32 {
            return Err(LgbmError::new(format!(
                "Invalid Dataset Index. Given: {} Max Allowed: {}",
                dataset_index,
                self.validation_data.len()
            )));
        }
        let names = ffi::get_eval_names(self.handle)?;
        let scores = ffi::get_eval_scores(self.handle, dataset_index, names.len())?;
        Ok(names
            .into_iter()
            .zip(scores)
            .map(|(metric_name, score)| EvalResult { metric_name, score })
            .collect())
    }

    /// this should take &mut self, because it changes the model
    fn train_loop(&mut self, max_iterations: i32) -> Result<(), LgbmError> {
        let mut is_finished = 0;
        let mut i = 0;
        while is_finished == 0 && i < max_iterations {
            // callback stuff here
            ffi::train_one_step(self.handle, &mut is_finished)?;
            i += 1;
        }
        Ok(())
    }

    /*   /// Train a booster further with a new dataset.
     /// This should not reset the already existing submodels.
     /// Pass an empty array as validation data, if you don't want to validate the train results.
     /// TODO validate this after implemented
    pub fn finetune(
         &mut self,
         _train_data: DataSet,
         _validation_data: Vec<DataSet>,
     ) -> Result<(), LgbmError> {

     }*/
}

impl Drop for Booster {
    fn drop(&mut self) {
        ffi::free_booster(self.handle).expect("Something went wrong dropping the Booster.");
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn simple() {}
}
