use std::error::Error;
use std::ffi::CString;

use lightgbm_sys::DatasetHandle;

use dataset::LoadedDataSet;
use {InputMatrix, OutputVec};

type FfiError = crate::Error;

pub(crate) fn drop_dataset(handle: DatasetHandle) -> Result<(), FfiError> {
    lgbm_call!(lightgbm_sys::LGBM_DatasetFree(handle))?;
    Ok(())
}

pub(crate) fn load_dataset_from_file(
    file_path: &str,
    dataset_params: &str,
    reference_dataset: &Option<DatasetHandle>,
) -> Result<DatasetHandle, FfiError> {
    let file_path_str = CString::new(file_path).unwrap();
    let params = CString::new(dataset_params).unwrap();
    let mut handle = std::ptr::null_mut();

    let reference = match reference_dataset {
        Some(h) => h.clone(),
        None => std::ptr::null_mut(),
    };

    lgbm_call!(lightgbm_sys::LGBM_DatasetCreateFromFile(
        file_path_str.as_ptr() as *const c_char,
        params.as_ptr() as *const c_char,
        reference,
        &mut handle
    ))?;

    Ok(handle)
}

pub(crate) fn load_from_vec(
    data: &InputMatrix,
    label: &OutputVec,
    dataset_params: &str,
    reference_dataset: &Option<DatasetHandle>,
) -> Result<DatasetHandle, FfiError> {
    let data_length = data.len();
    let feature_length = data[0].len();
    let params = CString::new(dataset_params).unwrap();
    let label_str = CString::new("label").unwrap();

    let reference = match reference_dataset {
        Some(h) => h.clone(),
        None => std::ptr::null_mut(),
    };

    let mut handle = std::ptr::null_mut();
    // mhhh..... does lightgbm reserve new space or uses this one
    let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

    if data_length > i32::MAX as usize || feature_length > i32::MAX as usize {
        return Err(FfiError::new(format!(
            "received old_dataset of size {}x{}, but at most {}x{} is supported",
            data_length,
            feature_length,
            i32::MAX,
            i32::MAX
        )));
    }

    lgbm_call!(lightgbm_sys::LGBM_DatasetCreateFromMat(
        flat_data.as_ptr() as *const c_void,
        lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
        data_length as i32,
        feature_length as i32,
        1_i32,
        params.as_ptr() as *const c_char,
        reference,
        &mut handle
    ))?;

    lgbm_call!(lightgbm_sys::LGBM_DatasetSetField(
        handle,
        label_str.as_ptr() as *const c_char,
        label.as_ptr() as *const c_void,
        data_length as i32,
        lightgbm_sys::C_API_DTYPE_FLOAT32 as i32
    ))?;

    Ok(handle)
}
