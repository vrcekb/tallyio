//! Build script for `strategy_core` crate
//!
//! Optimizes compilation for AMD EPYC 9454P architecture and enables
//! performance-critical features for `TallyIO` strategy execution.

use std::{env, process::Command, time::{SystemTime, UNIX_EPOCH}};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");

    // Get target architecture
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Configure for AMD EPYC 9454P (znver3 architecture)
    if target_arch == "x86_64" {
        // Configure x86_64 specific optimizations for AMD EPYC
        println!("cargo:rustc-env=TARGET_CPU=znver3");

        // Enable AMD EPYC specific features via RUSTFLAGS instead of cfg
        // This avoids the explicit_builtin_cfgs_in_flags warning
        println!("cargo:rustc-env=TARGET_CPU_FEATURES=avx2,fma,bmi1,bmi2,lzcnt,popcnt,sse4.2,aes,pclmulqdq");

        // Enable AVX-512 if feature is enabled
        if env::var("CARGO_FEATURE_AVX512").is_ok() {
            println!("cargo:rustc-env=AVX512_FEATURES=avx512f,avx512vl,avx512bw,avx512dq,avx512cd");
        }
    }

    // Configure OS-specific optimizations
    match target_os.as_str() {
        "linux" => {
            // Configure Linux-specific optimizations
            println!("cargo:rustc-link-lib=numa");
            println!("cargo:rustc-link-lib=pthread");

            // Enable NUMA awareness
            println!("cargo:rustc-cfg=numa_aware");

            // Enable high-resolution timers
            println!("cargo:rustc-cfg=high_res_timer");
        }
        "windows" => {
            // Configure Windows-specific optimizations
            // Windows-specific optimizations for development environment
            println!("cargo:rustc-link-lib=kernel32");
            println!("cargo:rustc-link-lib=user32");
        }
        _ => {}
    }

    // Configure SIMD feature detection and optimization
    if env::var("CARGO_FEATURE_SIMD").is_ok() {
        println!("cargo:rustc-cfg=simd_enabled");

        // Enable runtime SIMD detection
        println!("cargo:rustc-cfg=runtime_simd_detection");

        // Configure SIMD width based on target
        println!("cargo:rustc-env=SIMD_WIDTH=512"); // AVX-512 width
    }

    // Configure memory allocation optimizations
    // Enable jemalloc for better memory allocation performance
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=jemalloc");
        println!("cargo:rustc-cfg=jemalloc_allocator");
    }

    // Configure cache line size for AMD EPYC
    println!("cargo:rustc-env=CACHE_LINE_SIZE=64");

    // Configure NUMA topology
    println!("cargo:rustc-env=NUMA_NODES=2"); // Typical for EPYC systems

    // Generate build information for runtime diagnostics
    // Generate build timestamp
    let build_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={build_time}");

    // Generate git commit hash if available
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if output.status.success() {
            let commit_hash_raw = String::from_utf8_lossy(&output.stdout);
            let commit_hash = commit_hash_raw.trim();
            println!("cargo:rustc-env=GIT_COMMIT_HASH={commit_hash}");
        }
    }

    // Generate build profile information
    let profile = env::var("PROFILE").unwrap_or_default();
    println!("cargo:rustc-env=BUILD_PROFILE={profile}");

    // Generate optimization level
    let opt_level = env::var("OPT_LEVEL").unwrap_or_default();
    println!("cargo:rustc-env=OPT_LEVEL={opt_level}");

    // Generate target triple
    let target = env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=BUILD_TARGET={target}");
}

// All functions have been inlined into main() to avoid single_call_fn warnings
