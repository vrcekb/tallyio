//! Build script for `hot_path` crate with CPU feature detection and optimization.
//!
//! This build script detects CPU features at compile time and enables
//! appropriate optimizations for AMD EPYC 9454P processors.

#![expect(
    clippy::blanket_clippy_restriction_lints,
    reason = "Ultra-strict clippy profile requires individual restriction lints"
)]

use core::fmt::Write as _;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

/// AMD EPYC 9454P specific CPU features we want to detect and enable
const TARGET_CPU_FEATURES: &[&str] = &[
    "avx2",      // Advanced Vector Extensions 2
    "avx512f",   // AVX-512 Foundation
    "avx512vl",  // AVX-512 Vector Length Extensions
    "avx512bw",  // AVX-512 Byte and Word Instructions
    "avx512dq",  // AVX-512 Doubleword and Quadword Instructions
    "avx512cd",  // AVX-512 Conflict Detection Instructions
    "fma",       // Fused Multiply-Add
    "bmi1",      // Bit Manipulation Instruction Set 1
    "bmi2",      // Bit Manipulation Instruction Set 2
    "popcnt",    // Population Count
    "lzcnt",     // Leading Zero Count
    "adx",       // Multi-precision Add-Carry Instruction Extensions
    "aes",       // AES instruction set
    "pclmulqdq", // Carry-less Multiplication
    "sha",       // SHA extensions
    "rdrand",    // Hardware random number generator
    "rdseed",    // Hardware random seed generator
];

/// Target CPU for AMD EPYC 9454P (Zen 4 architecture)
const TARGET_CPU: &str = "znver4";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");

    // Only apply optimizations for x86_64 targets
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "x86_64" {
        println!("cargo:warning=Hot path optimizations are only available for x86_64 targets");
        return;
    }

    // Detect and configure CPU features
    configure_cpu_features();

    // Configure target-specific optimizations
    configure_target_optimizations();

    // Generate CPU feature detection code
    generate_feature_detection();

    // Configure link-time optimizations
    configure_lto();

    // Validate production requirements if needed
    if is_production_target() {
        validate_production_requirements();
    }

    println!("cargo:rustc-cfg=hot_path_optimized");
}

/// Configure CPU-specific features and optimizations
#[expect(clippy::single_call_fn, reason = "Function is called once during build process")]
fn configure_cpu_features() {
    let mut rustflags = Vec::with_capacity(10);

    // Set target CPU for optimal instruction selection
    rustflags.push(format!("-C target-cpu={TARGET_CPU}"));

    // Enable specific CPU features
    let features = TARGET_CPU_FEATURES.join(",+");
    rustflags.push(format!("-C target-feature=+{features}"));

    // Enable additional optimizations
    rustflags.extend_from_slice(&[
        "-C opt-level=3".to_owned(),
        "-C codegen-units=1".to_owned(),
        "-C lto=fat".to_owned(),
        "-C panic=abort".to_owned(),
        "-C embed-bitcode=yes".to_owned(),
        "-C debug-assertions=off".to_owned(),
        "-C overflow-checks=off".to_owned(),
    ]);

    // Apply RUSTFLAGS
    for flag in rustflags {
        println!("cargo:rustc-env=RUSTFLAGS={flag}");
    }

    // Enable feature flags based on detected capabilities
    for feature in TARGET_CPU_FEATURES {
        println!("cargo:rustc-cfg=feature=\"{feature}\"");
        println!("cargo:rustc-cfg=has_{feature}");
    }
}

/// Configure target-specific optimizations
#[expect(clippy::single_call_fn, reason = "Function is called once during build process")]
fn configure_target_optimizations() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target_os.as_str() {
        "linux" => {
            // Linux-specific optimizations for Ubuntu 22.04 LTS
            println!("cargo:rustc-cfg=target_linux");
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
            println!("cargo:rustc-link-arg=-Wl,--gc-sections");

            // Enable NUMA awareness
            println!("cargo:rustc-cfg=numa_aware");
        }
        "windows" => {
            // Windows development environment
            println!("cargo:rustc-cfg=target_windows");
        }
        _ => {
            println!("cargo:warning=Unsupported target OS: {target_os}");
        }
    }
}

/// Generate CPU feature detection code at compile time
#[expect(clippy::single_call_fn, reason = "Function is called once during build process")]
#[expect(clippy::print_stderr, reason = "Build script needs to report errors to stderr")]
#[expect(clippy::exit, reason = "Build script must exit on critical errors")]
fn generate_feature_detection() {
    let out_dir = env::var("OUT_DIR").unwrap_or_else(|_| {
        eprintln!("OUT_DIR environment variable not set");
        process::exit(1);
    });
    let dest_path = PathBuf::from(out_dir).join("cpu_features.rs");
    
    let mut code = String::with_capacity(6000);
    code.push_str("//! Generated CPU feature detection code\n\n");
    code.push_str("use core::sync::atomic::{AtomicBool, Ordering};\n\n");
    
    // Generate static feature flags
    for feature in TARGET_CPU_FEATURES {
        writeln!(
            &mut code,
            "static HAS_{}: AtomicBool = AtomicBool::new(false);",
            feature.to_uppercase()
        ).unwrap_or_else(|_| {
            eprintln!("Failed to write feature flag");
            process::exit(1);
        });
    }
    
    code.push_str("\n/// Initialize CPU feature detection\n");
    code.push_str("pub fn init_cpu_features() {\n");
    
    // Generate feature detection for each CPU feature
    for feature in TARGET_CPU_FEATURES {
        writeln!(
            &mut code,
            "    #[cfg(target_arch = \"x86_64\")]\n    {{\n        if is_x86_feature_detected!(\"{feature}\") {{\n            HAS_{}.store(true, Ordering::Relaxed);\n        }}\n    }}",
            feature.to_uppercase()
        ).unwrap_or_else(|_| {
            eprintln!("Failed to write feature detection");
            process::exit(1);
        });
    }
    
    code.push_str("}\n\n");
    
    // Generate feature check functions
    for feature in TARGET_CPU_FEATURES {
        writeln!(
            &mut code,
            "/// Check if {} is available\n#[inline(always)]\npub fn has_{feature}() -> bool {{\n    HAS_{}.load(Ordering::Relaxed)\n}}",
            feature.to_uppercase(),
            feature.to_uppercase()
        ).unwrap_or_else(|_| {
            eprintln!("Failed to write feature check function");
            process::exit(1);
        });
        writeln!(&mut code).unwrap_or_else(|_| {
            eprintln!("Failed to write newline");
            process::exit(1);
        });
    }
    
    // Generate optimized function selection
    code.push_str("/// Select optimal implementation based on CPU features\n");
    code.push_str("#[inline(always)]\n");
    code.push_str("pub fn select_optimal_impl() -> &'static str {\n");
    code.push_str("    if has_avx512f() && has_avx512vl() {\n");
    code.push_str("        \"avx512\"\n");
    code.push_str("    } else if has_avx2() {\n");
    code.push_str("        \"avx2\"\n");
    code.push_str("    } else {\n");
    code.push_str("        \"scalar\"\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    
    fs::write(&dest_path, code).unwrap_or_else(|_| {
        eprintln!("Failed to write CPU features file");
        process::exit(1);
    });

    println!("cargo:rustc-env=CPU_FEATURES_PATH={}", dest_path.display());
}

/// Configure link-time optimizations
#[expect(clippy::single_call_fn, reason = "Function is called once during build process")]
fn configure_lto() {
    // Enable thin LTO for faster builds in debug mode
    if env::var("PROFILE").unwrap_or_default() == "debug" {
        println!("cargo:rustc-lto=thin");
    } else {
        // Full LTO for release builds
        println!("cargo:rustc-lto=fat");
    }
    
    // Enable cross-language LTO if using C/C++ code
    println!("cargo:rustc-link-arg=-flto");
    
    // Optimize for size and speed
    println!("cargo:rustc-link-arg=-O3");
    println!("cargo:rustc-link-arg=-march=znver4");
    
    // Enable additional linker optimizations
    println!("cargo:rustc-link-arg=-Wl,-O3");
    println!("cargo:rustc-link-arg=-Wl,--icf=all");
}

/// Check if we're building for the target production environment
#[expect(clippy::implicit_return, reason = "Explicit return not needed for simple boolean expression")]
fn is_production_target() -> bool {
    let target_triple = env::var("TARGET").unwrap_or_default();
    target_triple.contains("x86_64") && target_triple.contains("linux")
}

/// Validate that we have the required CPU features for production
#[expect(clippy::single_call_fn, reason = "Function is called once during build process")]
fn validate_production_requirements() {
    if !is_production_target() {
        return;
    }

    // Ensure we have the minimum required features for production
    let required_features = ["avx2", "fma", "bmi2", "popcnt"];

    for feature in required_features {
        assert!(
            TARGET_CPU_FEATURES.contains(&feature),
            "Required CPU feature '{feature}' not available for production build"
        );
    }

    println!("cargo:rustc-cfg=production_validated");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_cpu_features() {
        assert!(!TARGET_CPU_FEATURES.is_empty());
        assert!(TARGET_CPU_FEATURES.contains(&"avx2"));
        assert!(TARGET_CPU_FEATURES.contains(&"avx512f"));
    }

    #[test]
    fn test_target_cpu() {
        assert_eq!(TARGET_CPU, "znver4");
    }
}
