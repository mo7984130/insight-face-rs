use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    FaceDetector, FaceRecognizer, Result, config::common::IDLE_SLEEP_DURATION, types::Face,
};
use image::RgbImage;
use tracing::error;

#[derive(Debug, Clone)]
pub struct FaceEngineConfig {
    det_model_path: PathBuf,
    rec_model_path: PathBuf,
    idle_timeout: Duration,
}

impl FaceEngineConfig {
    pub fn new(
        det_model_path: impl Into<PathBuf>,
        rec_model_path: impl Into<PathBuf>,
        idle_timeout: Duration,
    ) -> Self {
        Self {
            det_model_path: det_model_path.into(),
            rec_model_path: rec_model_path.into(),
            idle_timeout,
        }
    }
}

struct FaceState {
    det: Option<FaceDetector>,
    rec: Option<FaceRecognizer>,
    last_used: Option<Instant>,
}
pub struct FaceEngine {
    config: FaceEngineConfig,
    state: Mutex<FaceState>,
}
impl FaceEngine {
    pub fn new(config: &FaceEngineConfig) -> Result<Self> {
        let this = Self {
            config: config.clone(),
            state: Mutex::new(FaceState {
                det: None,
                rec: None,
                last_used: None,
            }),
        };
        this.load()?;
        Ok(this)
    }

    pub fn new_without_load(config: &FaceEngineConfig) -> Self {
        Self {
            config: config.clone(),
            state: Mutex::new(FaceState {
                det: None,
                rec: None,
                last_used: None,
            }),
        }
    }

    fn ensure_loaded_locked(&self, state: &mut FaceState) -> Result<()> {
        self.load_det_locked(state)?;
        self.load_rec_locked(state)?;
        Ok(())
    }

    pub fn run(&self, img: &RgbImage) -> Result<Vec<Face>> {
        let mut state = self.state.lock()?;
        self.ensure_loaded_locked(&mut state)?;

        let faces = state.det.as_mut().unwrap().detect(img)?;

        let embeddings = state.rec.as_mut().unwrap().extract_embedding(img, &faces)?;

        let results = faces
            .into_iter()
            .zip(embeddings)
            .map(|(face, embedding)| Face::from(face, embedding))
            .collect();

        state.last_used = Some(Instant::now());

        Ok(results)
    }

    pub fn run_from_file(&self, path: impl AsRef<Path>) -> Result<Vec<Face>> {
        let img = image::open(path)?.to_rgb8();
        self.run(&img)
    }

    fn load_det_locked(&self, state: &mut FaceState) -> Result<()> {
        if state.det.is_none() {
            state.det = Some(FaceDetector::new(
                &self.config.det_model_path,
                None,
                None,
                None,
            )?);
        }
        Ok(())
    }

    fn load_rec_locked(&self, state: &mut FaceState) -> Result<()> {
        if state.rec.is_none() {
            state.rec = Some(FaceRecognizer::new(&self.config.rec_model_path, None)?);
        }
        Ok(())
    }

    pub fn load(&self) -> Result<()> {
        let mut state = self.state.lock()?;
        self.ensure_loaded_locked(&mut state)?;
        Ok(())
    }

    pub fn unload(&self) -> Result<()> {
        let mut state = self.state.lock()?;
        self.unload_locked(&mut state);
        Ok(())
    }

    fn unload_locked(&self, state: &mut FaceState) {
        state.det.take();
        state.rec.take();

        state.last_used = None;

        #[cfg(target_os = "linux")]
        unsafe {
            libc::malloc_trim(0);
        }
    }

    pub fn reclaim_if_idle(&self) -> Result<()> {
        if self.config.idle_timeout.is_zero() {
            return Ok(());
        }

        let mut state = self.state.lock()?;
        let Some(last_used) = state.last_used else {
            return Ok(());
        };

        if last_used.elapsed() >= self.config.idle_timeout {
            self.unload_locked(&mut state);
            state.last_used = None;
        }

        Ok(())
    }

    pub fn start_reaper_thread(engine: &Arc<Self>) {
        let weak = Arc::downgrade(engine);

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(IDLE_SLEEP_DURATION);

                let Some(engine) = weak.upgrade() else {
                    break;
                };

                if let Err(err) = engine.reclaim_if_idle() {
                    error!(error = %err, "face engine reaper stopped");
                    break;
                };
            }
        });
    }

    #[cfg(feature = "tokio")]
    pub fn start_reaper(engine: &Arc<Self>) {
        let weak = Arc::downgrade(engine);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(IDLE_SLEEP_DURATION);

            interval.tick().await;

            loop {
                interval.tick().await;

                let Some(engine) = weak.upgrade() else {
                    break;
                };

                if let Err(err) = engine.reclaim_if_idle() {
                    error!(error = %err, "face engine reaper stopped");
                    break;
                };
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_idle_timeout_does_not_reclaim_models() {
        let config = FaceEngineConfig::new("det.onnx", "rec.onnx", Duration::ZERO);
        let engine = FaceEngine::new_without_load(&config);
        engine.state.lock().unwrap().last_used = Some(Instant::now());

        engine.reclaim_if_idle().unwrap();

        assert!(engine.state.lock().unwrap().last_used.is_some());
    }
}
