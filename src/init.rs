use crate::Error;

#[cfg(feature = "load-dynamic")]
static INIT: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();

#[cfg(feature = "load-dynamic")]
pub(crate) fn init_ort() -> Result<(), Error> {
    use std::path::Path;

    let result = INIT.get_or_init(|| {
        let path = "/opt/onnxruntime/libonnxruntime.so";

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

#[cfg(not(feature = "load-dynamic"))]
pub(crate) fn init_ort() -> Result<(), Error> {
    Ok(())
}
