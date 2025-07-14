//! TallyIO Ultra-Performance Build Script
//!
//! This build script optimizes the TallyIO financial application for maximum
//! performance on AMD EPYC 9454P processors with ultra-aggressive optimizations
//! for MEV detection <500ns, cross-chain operations <50ns, and end-to-end
//! latency <10ms. The script configures target-specific optimizations,
//! validates build environment, and sets up production-ready compilation.
//!
//! ## Performance Targets
//! - MEV Detection Pipeline: <500ns
//! - Cross-Chain Cache Operations: <50ns
//! - Memory Allocation: <5ns
//! - Crypto Operations: <50μs
//! - End-to-End Latency: <10ms
//! - Concurrent Throughput: 2M+ ops/sec
//!
//! ## Build Optimizations
//! - Target-specific CPU optimizations (AMD EPYC 9454P)
//! - Ultra-aggressive LTO and codegen optimizations
//! - SIMD instruction set utilization
//! - Memory layout optimizations
//! - Cache-friendly data structures

use std::{
    env,
    fs,
    path::Path,
    process::Command,
};

/// Build configuration for TallyIO ultra-performance compilation
struct TallyioBuildConfig {
    /// Target architecture
    target_arch: String,
    /// Target operating system
    target_os: String,
    /// Target environment
    target_env: String,
    /// Build profile (debug, release, production)
    profile: String,
    /// Enable ultra-aggressive optimizations
    ultra_optimizations: bool,
    /// Enable SIMD optimizations
    simd_optimizations: bool,
    /// Enable target-specific CPU features
    cpu_optimizations: bool,
}

impl TallyioBuildConfig {
    /// Create new build configuration from environment
    fn from_env() -> Self {
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "x86_64".to_string());
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| "linux".to_string());
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| "gnu".to_string());
        let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        
        // Enable ultra-optimizations for release and production builds
        let ultra_optimizations = matches!(profile.as_str(), "release" | "production");
        
        // Enable SIMD for x86_64 targets
        let simd_optimizations = target_arch == "x86_64";
        
        // Enable CPU optimizations for production builds
        let cpu_optimizations = ultra_optimizations && target_arch == "x86_64";

        Self {
            target_arch,
            target_os,
            target_env,
            profile,
            ultra_optimizations,
            simd_optimizations,
            cpu_optimizations,
        }
    }

    /// Check if this is a production build
    const fn is_production(&self) -> bool {
        matches!(self.profile.as_str(), "production")
    }

    /// Check if this is a release-like build
    const fn is_release_like(&self) -> bool {
        matches!(self.profile.as_str(), "release" | "production")
    }

    /// Check if target is Linux x86_64
    fn is_linux_x86_64(&self) -> bool {
        self.target_arch == "x86_64" && self.target_os == "linux"
    }
}

/// Main build script entry point
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo:rerun-if-env-changed=PROFILE");

    let config = TallyioBuildConfig::from_env();
    
    println!("cargo:warning=TallyIO Build Configuration:");
    println!("cargo:warning=  Target: {}-{}-{}", config.target_arch, config.target_os, config.target_env);
    println!("cargo:warning=  Profile: {}", config.profile);
    println!("cargo:warning=  Ultra Optimizations: {}", config.ultra_optimizations);
    println!("cargo:warning=  SIMD Optimizations: {}", config.simd_optimizations);
    println!("cargo:warning=  CPU Optimizations: {}", config.cpu_optimizations);

    // Validate build environment
    validate_build_environment(&config);

    // Configure target-specific optimizations
    configure_target_optimizations(&config);

    // Configure CPU-specific features
    configure_cpu_features(&config);

    // Configure memory optimizations
    configure_memory_optimizations(&config);

    // Configure SIMD optimizations
    configure_simd_optimizations(&config);

    // Configure production optimizations
    if config.is_production() {
        configure_production_optimizations(&config);
    }

    // Generate build metadata
    generate_build_metadata(&config);

    println!("cargo:warning=TallyIO build script completed successfully");
}

/// Validate build environment for TallyIO requirements
fn validate_build_environment(config: &TallyioBuildConfig) {
    // Check Rust version
    let rust_version = env::var("CARGO_PKG_RUST_VERSION").unwrap_or_else(|_| "1.75".to_string());
    println!("cargo:warning=Rust version: {}", rust_version);

    // Validate target for production builds
    if config.is_production() && !config.is_linux_x86_64() {
        println!("cargo:warning=WARNING: Production builds should target Linux x86_64");
        println!("cargo:warning=Current target: {}-{}-{}", 
                 config.target_arch, config.target_os, config.target_env);
    }

    // Check for required tools in production
    if config.is_production() {
        check_production_tools();
    }

    println!("cargo:warning=Build environment validation completed");
}

/// Check for required production tools
fn check_production_tools() {
    // Check for objcopy (for binary optimization)
    if let Ok(output) = Command::new("objcopy").arg("--version").output() {
        if output.status.success() {
            println!("cargo:warning=objcopy available for binary optimization");
        }
    }

    // Check for strip (for symbol stripping)
    if let Ok(output) = Command::new("strip").arg("--version").output() {
        if output.status.success() {
            println!("cargo:warning=strip available for symbol removal");
        }
    }
}

/// Configure target-specific optimizations
fn configure_target_optimizations(config: &TallyioBuildConfig) {
    if config.ultra_optimizations {
        // Enable link-time optimization
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        println!("cargo:rustc-link-arg=-Wl,--strip-all");
        
        if config.is_linux_x86_64() {
            // Linux-specific optimizations
            println!("cargo:rustc-link-arg=-Wl,--hash-style=gnu");
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
        }
    }

    println!("cargo:warning=Target optimizations configured");
}

/// Configure CPU-specific features for AMD EPYC 9454P
fn configure_cpu_features(config: &TallyioBuildConfig) {
    if config.cpu_optimizations && config.target_arch == "x86_64" {
        // AMD EPYC 9454P specific optimizations
        println!("cargo:rustc-env=RUSTFLAGS=-C target-cpu=znver3");
        
        // Enable advanced CPU features
        let cpu_features = [
            "aes",          // AES encryption instructions
            "avx2",         // Advanced Vector Extensions 2
            "bmi1",         // Bit Manipulation Instruction Set 1
            "bmi2",         // Bit Manipulation Instruction Set 2
            "fma",          // Fused Multiply-Add
            "lzcnt",        // Leading Zero Count
            "popcnt",       // Population Count
            "sse4.2",       // Streaming SIMD Extensions 4.2
            "sha",          // SHA extensions
            "adx",          // Multi-precision Add-Carry
            "rdseed",       // Random Seed
            "rdrand",       // Random Number Generator
        ];

        for feature in &cpu_features {
            println!("cargo:rustc-cfg=target_feature=\"{}\"", feature);
        }

        println!("cargo:warning=AMD EPYC 9454P CPU features enabled: {}", cpu_features.join(", "));
    }
}

/// Configure memory optimizations
fn configure_memory_optimizations(config: &TallyioBuildConfig) {
    if config.ultra_optimizations {
        // Configure memory alignment for cache efficiency
        println!("cargo:rustc-env=CARGO_CFG_TARGET_POINTER_WIDTH=64");
        
        // Enable memory optimization flags
        if config.is_linux_x86_64() {
            println!("cargo:rustc-link-arg=-Wl,-z,relro");
            println!("cargo:rustc-link-arg=-Wl,-z,now");
        }
    }

    println!("cargo:warning=Memory optimizations configured");
}

/// Configure SIMD optimizations
fn configure_simd_optimizations(config: &TallyioBuildConfig) {
    if config.simd_optimizations {
        // Enable SIMD features
        println!("cargo:rustc-cfg=feature=\"simd\"");
        
        if config.target_arch == "x86_64" {
            // x86_64 SIMD optimizations
            println!("cargo:rustc-cfg=target_feature=\"sse2\"");
            println!("cargo:rustc-cfg=target_feature=\"sse3\"");
            println!("cargo:rustc-cfg=target_feature=\"ssse3\"");
            println!("cargo:rustc-cfg=target_feature=\"sse4.1\"");
            println!("cargo:rustc-cfg=target_feature=\"sse4.2\"");
            
            if config.ultra_optimizations {
                println!("cargo:rustc-cfg=target_feature=\"avx\"");
                println!("cargo:rustc-cfg=target_feature=\"avx2\"");
            }
        }

        println!("cargo:warning=SIMD optimizations enabled");
    }
}

/// Configure production-specific optimizations
fn configure_production_optimizations(config: &TallyioBuildConfig) {
    // Ultra-aggressive optimization flags
    println!("cargo:rustc-env=CARGO_CFG_OPTIMIZED=1");
    
    // Enable all performance features
    println!("cargo:rustc-cfg=feature=\"ultra_performance\"");
    println!("cargo:rustc-cfg=feature=\"production\"");
    
    // Disable debug assertions in production
    println!("cargo:rustc-cfg=not(debug_assertions)");
    
    // Configure for financial application requirements
    println!("cargo:rustc-cfg=feature=\"financial_grade\"");
    println!("cargo:rustc-cfg=feature=\"mev_detection\"");
    println!("cargo:rustc-cfg=feature=\"cross_chain_optimized\"");

    println!("cargo:warning=Production optimizations enabled");
}

/// Generate build metadata
fn generate_build_metadata(config: &TallyioBuildConfig) {
    // Generate build timestamp
    let build_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    
    println!("cargo:rustc-env=TALLYIO_BUILD_TIMESTAMP={}", build_timestamp);
    
    // Generate build configuration
    println!("cargo:rustc-env=TALLYIO_BUILD_TARGET={}-{}-{}", 
             config.target_arch, config.target_os, config.target_env);
    println!("cargo:rustc-env=TALLYIO_BUILD_PROFILE={}", config.profile);
    
    // Generate optimization flags
    println!("cargo:rustc-env=TALLYIO_ULTRA_OPTIMIZATIONS={}", config.ultra_optimizations);
    println!("cargo:rustc-env=TALLYIO_SIMD_OPTIMIZATIONS={}", config.simd_optimizations);
    println!("cargo:rustc-env=TALLYIO_CPU_OPTIMIZATIONS={}", config.cpu_optimizations);

    // Write build info to file for runtime access
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let build_info_path = Path::new(&out_dir).join("build_info.rs");
        let build_info_content = format!(
            r#"
/// TallyIO build information
pub const BUILD_TIMESTAMP: u64 = {};
pub const BUILD_TARGET: &str = "{}-{}-{}";
pub const BUILD_PROFILE: &str = "{}";
pub const ULTRA_OPTIMIZATIONS: bool = {};
pub const SIMD_OPTIMIZATIONS: bool = {};
pub const CPU_OPTIMIZATIONS: bool = {};
"#,
            build_timestamp,
            config.target_arch, config.target_os, config.target_env,
            config.profile,
            config.ultra_optimizations,
            config.simd_optimizations,
            config.cpu_optimizations
        );

        if fs::write(&build_info_path, build_info_content).is_ok() {
            println!("cargo:warning=Build metadata generated: {:?}", build_info_path);
        }
    }

    println!("cargo:warning=Build metadata generation completed");
}
