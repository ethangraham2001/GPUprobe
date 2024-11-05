mod gpuprobe;

use std::{net::SocketAddr, sync::Arc};

use clap::Parser;
use gpuprobe::Gpuprobe;
use prometheus_client::{encoding::text::encode, registry::Registry};

use axum::{
    extract::State, http::StatusCode, response::IntoResponse, routing::get, Router
};

use crate::gpuprobe::metrics::GpuprobeMetrics;

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

const INTERVAL_DURATION_SECONDS: u64 = 5;


// Explicitly define the state type
type SharedRegistry = Arc<Registry>;

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


    for _ in 0..=10 {
        gpuprobe.collect_metrics_uprobes().unwrap();
        let mut buff = String::new();
        encode(&mut buff, &registry).expect("encoding failure");
        println!("{buff}");

        println!("========================\n");
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    let arc_registry: Arc<Registry> = Arc::new(registry);

    // Create router with metrics endpoint
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(arc_registry);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:9091").await.unwrap();
    println!("Metrics server listening on http://0.0.0.0:9091/metrics");
    axum::serve(listener, app).await.unwrap();

    Ok(())

}

async fn metrics_handler(registry: State<Arc<Registry>>) -> impl IntoResponse {
    let mut buffer = String::new();
    match encode(&mut buffer, &registry) {
        Ok(()) => (StatusCode::OK, buffer),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, String::new()),
    }
}
