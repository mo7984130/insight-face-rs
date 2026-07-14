use std::path::Path;
use std::sync::OnceLock;

use crate::Error;
use crate::Result;

static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

pub(crate) fn init_ort() -> Result<()> {
    let result = INIT.get_or_init(|| {
        let path = "/opt/onnxruntime/libonnxruntime.so1";

        if !Path::new(path).is_file() {
            return Err(format!("onnxruntime dylib not found at {path}").to_string());
        }

        match ort::init_from(path) {
            Ok(b) => {
                b.commit();
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    });

    result
        .clone()
        .map_err(|e| Error::LoadLibError(e.to_string()))
}
