mod gpuprobe;

use std::{sync::Arc, time::Duration};

use tokio::{select, sync::Mutex};

use clap::Parser;
use gpuprobe::Gpuprobe;
use prometheus_client::{encoding::text::encode, registry::Registry};

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};

#[derive(Parser)]
#[command(author, version, about, long_about = None, arg_required_else_help = true)]
struct Args {
    /// Detects leaking calls to cudaMalloc from the CUDA runtime API.
    #[arg(long, exclusive = false)]
    memleak: bool,

    /// Maintains a histogram on frequencies of cuda kernel launches.
    #[arg(long, exclusive = false)]
    cudatrace: bool,

    /// Approximates bandwidth utilization of cudaMemcpy.
    #[arg(long, exclusive = false)]
    bandwidth_util: bool,
}

#[derive(Clone)]
struct AppState {
    gpuprobe: Arc<Mutex<Gpuprobe>>,
    registry: Arc<Registry>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let opts = gpuprobe::Opts {
        memleak: args.memleak,
        cudatrace: args.cudatrace,
        bandwidth_util: args.bandwidth_util,
    };

    let mut gpuprobe = gpuprobe::Gpuprobe::new(opts).unwrap();
    gpuprobe.attach_uprobes().unwrap();

    // Prometheus registry for exporting metrics
    let mut registry = Registry::default();
    gpuprobe.metrics.register(&mut registry);

    let registry: Arc<Registry> = Arc::new(registry);
    let gpuprobe = Mutex::new(gpuprobe);
    let gpuprobe = Arc::new(gpuprobe);

    // clones that are passed to the task that displays to stdout
    let gpuprobe_clone = Arc::clone(&gpuprobe);
    let registry_clone = Arc::clone(&registry);

    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(AppState { gpuprobe, registry });

    // a simple task that periodically displays metrics in their raw 
    // OpenMetrics format to stdout
    let stdout_handle = tokio::spawn(async move {
        loop {
            let mut probe = gpuprobe_clone.lock().await;
            probe.collect_metrics_uprobes().unwrap();

            let mut buff = String::new();
            encode(&mut buff, &registry_clone).unwrap();
            println!("{buff}");

            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9091").await.unwrap();
    let server_handle = axum::serve(listener, app);

    select! {
         _ = stdout_handle => {
            println!("Metrics printing task ended");
        }
        _ = server_handle => {
            println!("Server task ended");
        }
    }

    Ok(())
}

/// Handler for the endpoint that is scraped by Prometheus
async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state
        .gpuprobe
        .lock()
        .await
        .collect_metrics_uprobes()
        .unwrap();
    let mut buffer = String::new();
    match encode(&mut buffer, &state.registry) {
        Ok(()) => (StatusCode::OK, buffer),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, String::new()),
    }
}
