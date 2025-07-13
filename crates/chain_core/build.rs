//! Build script for chain-specific optimizations
//!
//! This build script performs compile-time optimizations and feature detection
//! for the chain core crate, optimized for AMD EPYC 9454P architecture.

use std::env;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Tell Cargo to rerun this build script if these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");

    // Get target information
    let target = env::var("TARGET").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // AMD EPYC 9454P specific optimizations
    if target_arch == "x86_64" {
        println!("cargo:rustc-cfg=target_cpu_epyc");

        // Enable AVX-512 for AMD EPYC 9454P
        if target.contains("linux") {
            println!("cargo:rustc-cfg=avx512_support");
        }

        // Enable SIMD optimizations
        println!("cargo:rustc-cfg=simd_optimizations");
        println!("cargo:rustc-cfg=sse4_2_support");
        println!("cargo:rustc-cfg=avx2_support");
        println!("cargo:rustc-cfg=fma_support");
        println!("cargo:rustc-cfg=cache_line_64");
    }

    // Linux-specific optimizations
    if target_os == "linux" {
        println!("cargo:rustc-cfg=linux_optimizations");
        println!("cargo:rustc-cfg=numa_support");
    }

    // Production vs development optimizations
    let profile = env::var("PROFILE").unwrap_or_default();
    if profile == "release" {
        println!("cargo:rustc-cfg=production_build");
    } else {
        println!("cargo:rustc-cfg=development_build");
    }

    // Check which chain features are enabled
    let features = [
        "ethereum", "bsc", "polygon", "arbitrum",
        "optimism", "base", "avalanche"
    ];

    for feature in &features {
        if env::var(format!("CARGO_FEATURE_{}", feature.to_uppercase())).is_ok() {
            println!("cargo:rustc-cfg=chain_{feature}");
        }
    }

    // Enable lock-free data structures
    println!("cargo:rustc-cfg=lock_free_structures");
    println!("cargo:rustc-cfg=memory_prefetch");

    // Generate build timestamp
    let build_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    println!("cargo:rustc-env=BUILD_TIMESTAMP={build_timestamp}");

    // Generate Git information if available
    if let Ok(git_hash) = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if git_hash.status.success() {
            let git_hash_str = String::from_utf8_lossy(&git_hash.stdout);
            let git_hash_trimmed = git_hash_str.trim();
            println!("cargo:rustc-env=GIT_HASH={git_hash_trimmed}");
        }
    }

    // Generate version information
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_default();
    println!("cargo:rustc-env=CRATE_VERSION={version}");

    // Generate target information
    println!("cargo:rustc-env=BUILD_TARGET={target}");

    // Generate optimization level
    let opt_level = env::var("OPT_LEVEL").unwrap_or_default();
    println!("cargo:rustc-env=OPT_LEVEL={opt_level}");

    // Generate profile information
    println!("cargo:rustc-env=BUILD_PROFILE={profile}");
}
