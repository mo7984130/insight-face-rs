use crate::Error;

#[cfg(feature = "load-dynamic")]
static INIT: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();

#[cfg(feature = "load-dynamic")]
pub(crate) fn init_ort() -> Result<(), Error> {
    let result = INIT.get_or_init(|| match ort::init_from("libonnxruntime.so") {
        Ok(b) => {
            b.commit();
            Ok(())
        }
        Err(e) => Err(e.to_string()),
    });

    result
        .clone()
        .map_err(|e| Error::LoadLibError(e.to_string()))
}

#[cfg(not(feature = "load-dynamic"))]
pub(crate) fn init_ort() -> Result<(), Error> {
    Ok(())
}
