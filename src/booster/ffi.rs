use std::ffi::CString;

use libc::{c_char, c_double, c_longlong, c_void};
use lightgbm_sys::{BoosterHandle, DatasetHandle};

use {lightgbm_sys, Matrixf64};

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
    data: &Matrixf64,
) -> Result<Matrixf64> {
    let data_length = data.len();
    let feature_length = data[0].len();
    let params = CString::new(prediction_params)?;
    let mut out_length: c_longlong = 0;
    let flat_data = data.clone().into_iter().flatten().collect::<Vec<_>>();
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

/// Get number of evaluation metrics
pub(crate) fn num_eval(handle: BoosterHandle) -> Result<i32> {
    let mut out_len = 0;
    lgbm_call!(lightgbm_sys::LGBM_BoosterGetEvalCounts(
        handle,
        &mut out_len
    ))?;
    Ok(out_len)
}

/// Get names of evaluation metrics
pub(crate) fn get_eval_names(handle: BoosterHandle) -> Result<Vec<String>> {
    let num_metrics = num_eval(handle)?;

    /////////////////////////////////////////////////////////////////////
    // call with 0-sized buffer to find out how much space to allocate
    /////////////////////////////////////////////////////////////////////
    let mut num_eval_names = 0;
    let mut out_buffer_len = 0;

    lgbm_call!(lightgbm_sys::LGBM_BoosterGetEvalNames(
        handle,
        0,
        &mut num_eval_names,
        0,
        &mut out_buffer_len,
        std::ptr::null_mut() as *mut *mut c_char
    ))?;

    /////////////////////////////////////////////////////////////////////
    // sanity check
    /////////////////////////////////////////////////////////////////////
    if num_eval_names != num_metrics {
        return Err(LgbmError::new(format!(
            "expected num_eval_names==num_metrics, but got {num_eval_names}!={num_metrics}. This is a bug in lightgbm or its rust wrapper"
        )));
    }

    /////////////////////////////////////////////////////////////////////
    // get the actual strings
    /////////////////////////////////////////////////////////////////////

    let mut out_strs = (0..num_metrics)
        .map(|_| (0..out_buffer_len).map(|_| 0).collect::<Vec<u8>>())
        .collect::<Vec<_>>();

    let mut out_strs_pointers = out_strs
        .iter_mut()
        .map(|s| s.as_mut_ptr())
        .collect::<Vec<_>>();

    let metric_name_length = out_buffer_len;

    lgbm_call!(lightgbm_sys::LGBM_BoosterGetEvalNames(
        handle,
        num_metrics,
        &mut num_eval_names,
        metric_name_length,
        &mut out_buffer_len,
        out_strs_pointers.as_mut_ptr() as *mut *mut c_char
    ))?;

    drop(out_strs_pointers); // don't let pointers outlive their target

    let mut output = Vec::with_capacity(out_strs.len());
    for mut out_str in out_strs {
        let first_null = out_str
            .iter()
            .enumerate()
            .find(|(_, e)| **e == 0)
            .map(|(i, _)| i)
            .expect("string not null terminated, possible memory corruption");
        out_str.truncate(first_null + 1);

        let string = CString::from_vec_with_nul(out_str)
            .expect("string memory invariant violated, possible memory corruption")
            .into_string()
            .map_err(|_| LgbmError::new("name not valid UTF-8"))?;
        output.push(string);
    }

    Ok(output)
}

pub(crate) fn get_eval_scores(
    handle: BoosterHandle,
    data_index: i32,
    num_metrics: usize,
) -> Result<Vec<f64>> {
    let mut out_len = 0;
    let out_result: Vec<f64> = vec![Default::default(); num_metrics];
    lgbm_call!(lightgbm_sys::LGBM_BoosterGetEval(
        handle,
        data_index,
        &mut out_len,
        out_result.as_ptr() as *mut c_double
    ))?;
    if out_len != out_result.len() as i32 {
        Err(LgbmError::new(
            "Output Array length doesn't match reported length.",
        ))
    } else {
        Ok(out_result)
    }
}

pub(crate) fn num_feature(booster: BoosterHandle) -> Result<i32> {
    let mut out_len = 0;
    lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumFeature(
        booster,
        &mut out_len
    ))?;
    Ok(out_len)
}
