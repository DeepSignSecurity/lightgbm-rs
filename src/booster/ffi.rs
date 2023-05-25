use std::ffi::CString;

use libc::{c_char, c_double, c_longlong, c_void};
use lightgbm_sys::{BoosterHandle, DatasetHandle};

use {lightgbm_sys, InputMatrix};

use crate::{LgbmError, Result};

pub(crate) fn new_booster(train_data: DatasetHandle, parsed_params: &str) -> Result<BoosterHandle> {
    let params_cstring = CString::new(parsed_params)?;
    let mut handle = std::ptr::null_mut();
    lgbm_call!(lightgbm_sys::LGBM_BoosterCreate(
        train_data,
        params_cstring.as_ptr() as *const c_char,
        &mut handle
    ))?;
    Ok(handle)
}

pub(crate) fn add_validation_data_to_booster(
    booster: BoosterHandle,
    validation_data_handle: DatasetHandle,
) -> Result<()> {
    lgbm_call!(lightgbm_sys::LGBM_BoosterAddValidData(
        booster,
        validation_data_handle
    ))
}

#[inline]
pub(crate) fn train_one_step(booster: BoosterHandle, is_finished: &mut i32) -> Result<()> {
    lgbm_call!(lightgbm_sys::LGBM_BoosterUpdateOneIter(
        booster,
        is_finished
    ))
}

pub(crate) fn get_num_classes(booster: BoosterHandle) -> Result<i32> {
    let mut num_classes = -1;
    lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumClasses(
        booster,
        &mut num_classes
    ))?;
    if num_classes > -1 {
        Ok(num_classes)
    } else {
        Err(LgbmError::new(
            "lgbm didn't update the number of classes correctly.",
        ))
    }
}

pub(crate) fn predict(
    booster: BoosterHandle,
    prediction_params: &str,
    data: InputMatrix,
) -> Result<InputMatrix> {
    let data_length = data.len();
    let feature_length = data[0].len();
    let params = CString::new(prediction_params)?;
    let mut out_length: c_longlong = 0;
    let flat_data = data.into_iter().flatten().collect::<Vec<_>>();
    let num_classes = get_num_classes(booster)?;
    let out_result: Vec<f64> = vec![Default::default(); data_length * num_classes as usize];

    lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMat(
        booster,
        flat_data.as_ptr() as *const c_void,
        lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
        data_length as i32,
        feature_length as i32,
        1_i32,
        0_i32,
        0_i32,
        -1_i32,
        params.as_ptr() as *const c_char,
        &mut out_length,
        out_result.as_ptr() as *mut c_double
    ))?;

    // reshape for multiclass [1,2,3,4,5,6] -> [[1,2,3], [4,5,6]]  # 3 class
    let reshaped_output = if num_classes > 1 {
        out_result
            .chunks(num_classes as usize)
            .map(|x| x.to_vec())
            .collect()
    } else {
        vec![out_result]
    };
    Ok(reshaped_output)
}

pub(crate) fn num_feature(booster: BoosterHandle) -> Result<i32> {
    let mut out_len = 0;
    lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumFeature(
        booster,
        &mut out_len
    ))?;
    Ok(out_len)
}
