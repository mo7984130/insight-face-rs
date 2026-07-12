use std::sync::Once;

static INIT: Once = Once::new();

pub(crate) fn ensure_backend_initialized() {
    INIT.call_once(|| {
        #[cfg(feature = "backend-tract")]
        ort::set_api(ort_tract::api());
    });
}
