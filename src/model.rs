use crate::Result;
use ort::{
    ep,
    session::{Session, SessionOutputs},
    value::{PrimitiveTensorElementType, Tensor},
};
use std::{fmt::Debug, path::Path};

pub(crate) struct OnnxModel {
    session: Session,
}

impl OnnxModel {
    pub(crate) fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ep::CUDA::default().build(), ep::CPU::default().build()])?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub(crate) fn run<T>(&mut self, input: Tensor<T>) -> Result<SessionOutputs<'_>>
    where
        T: PrimitiveTensorElementType + Clone + 'static + Debug,
    {
        let outputs = self.session.run(ort::inputs![input.into_dyn()])?;
        Ok(outputs)
    }
}
