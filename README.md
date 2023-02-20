# lightgbm-rs
LightGBM Rust binding

# About this Fork

This is an opinionated fork of [vaaaaanquish/lightgbm-rs](https://github.com/vaaaaanquish/lightgbm-rs) to fit needs at DeepSignSecurity. Generally we try to first offer fixes as PR to upstream, before merging them into `main` here. Exceptions apply for changes vaaaaanquish has already expressed no interest in, or that depend on other changes upstream hasn't accepted yet.

List of changes in `main`:

- [x] Merge vaaaaanquish/lightgbm-rs#35 (Fix truncated feature names)
- [x] Merge vaaaaanquish/lightgbm-rs#36 (add save_file_size api)
- [x] Merge vaaaaanquish/lightgbm-rs#37 (add load_string and save_string apis)
- [x] Merge vaaaaanquish/lightgbm-rs#43 (Set weights, but with more sanity checks and unit tests)
- [x] Switch to upstream microsoft/lightgbm
- [x] Merge vaaaaanquish/lightgbm-rs#44 (configure C++11 even more explicitly as suggested in vaaaaanquish/lightgbm-rs#40)

# Require

You need an environment that can build LightGBM.

```
# linux
apt install -y cmake libclang-dev libc++-dev gcc-multilib

# OS X
brew install cmake libomp
```

On Windows

1. Install CMake and VS Build Tools.
1. Install LLVM and set an environment variable `LIBCLANG_PATH` to PATH_TO_LLVM_BINARY (example: `C:\Program Files\LLVM\bin`)

Please see below for details.

- [LightGBM Installation-Guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

# Usage

Example LightGBM train.
```
extern crate serde_json;
use lightgbm::{Dataset, Booster};
use serde_json::json;

let data = vec![vec![1.0, 0.1, 0.2, 0.1],
               vec![0.7, 0.4, 0.5, 0.1],
               vec![0.9, 0.8, 0.5, 0.1],
               vec![0.2, 0.2, 0.8, 0.7],
               vec![0.1, 0.7, 1.0, 0.9]];
let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
let dataset = Dataset::from_mat(data, label).unwrap();
let params = json!{
   {
        "num_iterations": 3,
        "objective": "binary",
        "metric": "auc"
    }
};
let bst = Booster::train(dataset, &params).unwrap();
```

Please see the `./examples` for details.

|example|link|
|---|---|
|binary classification|[link](https://github.com/vaaaaanquish/lightgbm-rs/blob/main/examples/binary_classification/src/main.rs)|
|multiclass classification|[link](https://github.com/vaaaaanquish/lightgbm-rs/blob/main/examples/multiclass_classification/src/main.rs)|
|regression|[link](https://github.com/vaaaaanquish/lightgbm-rs/blob/main/examples/regression/src/main.rs)|



# Develop

```
git clone --recursive https://github.com/vaaaaanquish/lightgbm-rs
```

```
docker build -t lgbmrs .
docker run -it -v $PWD:/app lgbmrs bash

# cargo build
```


# Thanks

Much reference was made to implementation and documentation. Thanks.

- [microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost)
