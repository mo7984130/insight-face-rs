use ndarray::ShapeError;
use ort::session::builder::SessionBuilder;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("lib load error: {0}")]
    LoadLibError(String),
    #[error("model load error: {0}")]
    ModelLoadError(String),
    #[error("model run error: {0}")]
    ModelRunError(String),
    #[error("align face error: {0}")]
    AlignFaceError(String),
}

impl From<ort::Error<()>> for Error {
    fn from(err: ort::Error<()>) -> Self {
        Error::ModelLoadError(err.to_string())
    }
}

impl From<ort::Error<SessionBuilder>> for Error {
    fn from(err: ort::Error<SessionBuilder>) -> Self {
        Error::ModelLoadError(err.to_string())
    }
}

impl From<ShapeError> for Error {
    fn from(err: ShapeError) -> Self {
        Error::ModelRunError(err.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
