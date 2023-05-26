use booster::builder::{BoosterBuilder, ParamsMissing, TrainDataMissing};
use dataset::LoadedDataSet;
use LgbmError;
use Matrixf64;

mod builder;
mod ffi;

/// Evaluation Result of a Booster on a given Dataset.
/// Returned by get_eval
pub struct EvalResult {
    pub metric_name: String,
    pub score: f64,
}

/// Class that is returned by the builder, once fit() is called.
/// Used to interact with a trained booster.
pub struct Booster {
    handle: lightgbm_sys::BoosterHandle,
    #[allow(dead_code)]
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
    /// Output dimensions depend on booster task.
    /// Can return an Error if the input or model is corrupt.
    /// ```
    /// use lightgbm::booster::Booster;
    /// # use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// # let params = serde_json::json! {
    /// #             {
    /// #                 "num_iterations": 5,
    /// #                 "objective": "binary",
    /// #                 "metric": "auc",
    /// #                 "data_random_seed": 0
    /// #             }
    /// #         };
    /// # let x = vec![
    /// #             vec![1.0, 0.1, 0.2, 0.1],
    /// #             vec![0.7, 0.4, 0.5, 0.1],
    /// #             vec![0.9, 0.8, 0.5, 0.1],
    /// #             vec![0.2, 0.2, 0.8, 0.7],
    /// #             vec![0.1, 0.7, 1.0, 0.9]];
    /// # let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// # let train_data = DataSet::from_mat(x,y);
    /// # let input = vec![
    /// #             vec![8.0, 0.2, 0.4, 0.5],
    /// #             vec![0.9, 0.4, 0.3, 0.5],
    /// #             vec![0.5, 0.6, 0.3, 0.8],
    /// #             vec![0.244, 0.25, 0.9, 0.9],
    /// #             vec![0.4, 0.8, 0.8, 0.7],
    /// #         ];
    /// let booster = Booster::builder()
    ///     .add_train_data(train_data)     
    ///     .add_params(params)?
    ///     .fit()?;
    /// let pred = booster.predict(&input)?;
    ///
    /// assert_eq!(input.len(), pred[0].len()); // binary classification. One output value for each input vec
    /// # Ok(())}
    /// ```
    pub fn predict(&self, x: &Matrixf64) -> Result<Matrixf64, LgbmError> {
        let prediction_params = ""; // do we need this?
        self.predict_with_params(x, prediction_params)
    }

    /// Predict with additional params
    /// ```
    /// use lightgbm::booster::Booster;
    /// # use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// # let params = serde_json::json! {
    /// #             {
    /// #                 "num_iterations": 5,
    /// #                 "objective": "binary",
    /// #                 "metric": "auc",
    /// #                 "data_random_seed": 0
    /// #             }
    /// #         };
    /// # let x = vec![
    /// #             vec![1.0, 0.1, 0.2, 0.1],
    /// #             vec![0.7, 0.4, 0.5, 0.1],
    /// #             vec![0.9, 0.8, 0.5, 0.1],
    /// #             vec![0.2, 0.2, 0.8, 0.7],
    /// #             vec![0.1, 0.7, 1.0, 0.9]];
    /// # let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// # let train_data = DataSet::from_mat(x,y);
    /// # let input = vec![
    /// #             vec![8.0, 0.2, 0.4, 0.5],
    /// #             vec![0.9, 0.4, 0.3, 0.5],
    /// #             vec![0.5, 0.6, 0.3, 0.8],
    /// #             vec![0.244, 0.25, 0.9, 0.9],
    /// #             vec![0.4, 0.8, 0.8, 0.7],
    /// #         ];
    /// let booster = Booster::builder()
    ///     .add_train_data(train_data)     
    ///     .add_params(params)?
    ///     .fit()?;
    /// let predict_params = "predict_raw_score=true";
    /// let pred = booster.predict(&input)?;
    /// let pred_raw = booster.predict_with_params(&input, predict_params)?;
    ///
    /// # Ok(())}
    /// ```
    pub fn predict_with_params(
        &self,
        x: &Matrixf64,
        prediction_params: &str,
    ) -> Result<Matrixf64, LgbmError> {
        ffi::predict(self.handle, prediction_params, x)
    }

    /// Returns the scores for a certain dataset. You can use the index like this:
    /// 0 = Train Dataset
    /// 1 = 1. Validation Dataset
    /// 2 = 2. Validation Dataset
    /// ...
    /// n = nth Validation Dataset
    /// ```
    /// use lightgbm::booster::Booster;
    /// # use lightgbm::dataset::DataSet;
    /// # use lightgbm::LgbmError;
    ///
    /// # fn main() -> Result<(), LgbmError> {
    /// # let params = serde_json::json! {
    /// #             {
    /// #                 "num_iterations": 5,
    /// #                 "objective": "binary",
    /// #                 "metric": "auc",
    /// #                 "data_random_seed": 0
    /// #             }
    /// #         };
    /// # let x = vec![
    /// #             vec![1.0, 0.1, 0.2, 0.1],
    /// #             vec![0.7, 0.4, 0.5, 0.1],
    /// #             vec![0.9, 0.8, 0.5, 0.1],
    /// #             vec![0.2, 0.2, 0.8, 0.7],
    /// #             vec![0.1, 0.7, 1.0, 0.9]];
    /// # let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// # let train_data = DataSet::from_mat(x,y);
    /// # let input = vec![
    /// #             vec![8.0, 0.2, 0.4, 0.5],
    /// #             vec![0.9, 0.4, 0.3, 0.5],
    /// #             vec![0.5, 0.6, 0.3, 0.8],
    /// #             vec![0.244, 0.25, 0.9, 0.9],
    /// #             vec![0.4, 0.8, 0.8, 0.7],
    /// #         ];
    /// let booster = Booster::builder()
    ///     .add_train_data(train_data)     
    ///     .add_params(params)?
    ///     .fit()?;
    /// let pred = booster.predict(&input)?;
    /// let eval = booster.get_eval_result_for_dataset(0); // train data
    ///
    /// # Ok(())}
    /// ```
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
