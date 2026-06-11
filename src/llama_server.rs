use std::path::Path;
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct LlamaServer {
    pub port:  u16,
    /// Turns false when the child process exits (set by a background monitor thread).
    pub alive: Arc<AtomicBool>,
    child:     Arc<Mutex<Child>>,
}

impl LlamaServer {
    pub fn spawn(models_dir: &Path) -> anyhow::Result<Self> {
        let port = free_port()?;
        let cli  = crate::model_download::llamafile_path(models_dir);

        // On Linux, run via `sh` to bypass BINFMT_MISC/WSLInterop routing llamafile
        // through Windows on WSL2. `sh` uses the shell-script path in the APE polyglot,
        // which extracts and re-execs a native ELF from /tmp (only the server binary,
        // not the bundled model). Direct execution fails on WSL2 with WSLInterop enabled.
        #[cfg(target_os = "linux")]
        let mut cmd = {
            let mut c = std::process::Command::new("sh");
            c.arg(&cli);
            c
        };
        #[cfg(not(target_os = "linux"))]
        let mut cmd = std::process::Command::new(&cli);
        cmd.args([
            "--server",
            "--host", "127.0.0.1",
            "--port", &port.to_string(),
            "-c", "8192",
            "-t", "8",
        ]);

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }

        // Cosmopolitan's dlopen doesn't use ldconfig — it only searches LD_LIBRARY_PATH.
        // Prepend the standard CUDA/NVIDIA library directories so llamafile finds
        // libcuda.so.1 in Docker containers where LD_LIBRARY_PATH is otherwise empty.
        #[cfg(target_os = "linux")]
        {
            let arch = std::env::consts::ARCH; // "x86_64" or "aarch64"
            let extra = format!(
                "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/lib/{arch}-linux-gnu:/usr/lib/{arch}-linux-gnu"
            );
            let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let val = if existing.is_empty() { extra } else { format!("{extra}:{existing}") };
            cmd.env("LD_LIBRARY_PATH", val);
        }

        let child = cmd
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn llamafile: {e}"))?;

        let child  = Arc::new(Mutex::new(child));
        let alive  = Arc::new(AtomicBool::new(true));

        // Background OS thread: polls the child every second and sets `alive`
        // to false when it exits so the retry loop in summarize can fail fast.
        let child_mon = child.clone();
        let alive_mon = alive.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                match child_mon.lock().unwrap().try_wait() {
                    Ok(Some(s)) => {
                        alive_mon.store(false, Ordering::Release);
                        if s.success() {
                            tracing::info!("llamafile server exited cleanly");
                        } else {
                            tracing::warn!("llamafile server exited with status {s}");
                        }
                        break;
                    }
                    Ok(None) => {} // still running
                    Err(e)   => { tracing::warn!("llamafile wait error: {e}"); break; }
                }
            }
        });

        tracing::info!(port, "llamafile server spawned (model loading in background…)");
        Ok(Self { port, alive, child })
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let mut c = self.child.lock().unwrap();
        let _ = c.kill();
        let _ = c.wait();
    }
}

fn free_port() -> anyhow::Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}
