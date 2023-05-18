use booster::builder::{BoosterBuilder, ParamsNotAdded, TrainDataNotAdded};
use dataset::DataSet;
use Error;
use {InputMatrix, OutputVec};

mod builder;
mod ffi;

pub struct Booster {
    handle: lightgbm_sys::BoosterHandle,
    train_data: DataSet,
    validation_data: Vec<DataSet>,
}

impl Booster {
    /// Returns a builder. At least training data and params need to be added,
    /// so that the model can be fitted (built).
    pub fn builder() -> BoosterBuilder<TrainDataNotAdded, ParamsNotAdded> {
        BoosterBuilder::default()
    }

    /// Generates a prediction for a given Input.
    ///
    /// Can return an Error if the input or model is corrupt.
    pub fn predict(&self, x: &InputMatrix) -> Result<OutputVec, Error> {
        let _ = x[0][0] + 1_f64; // silence warning for now
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use booster::Booster;

    #[test]
    fn simple() {}
}
