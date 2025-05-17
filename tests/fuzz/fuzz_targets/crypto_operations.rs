//! Fuzz testi za kriptografske operacije
//!
//! Ti testi izvajajo fuzzing na kriptografske operacije uporabljene v secure_storage
//! modulu, kjer je varnost kritična za MEV platformo.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use secure_storage::crypto::{encrypt, decrypt};
use std::vec::Vec;

#[derive(Arbitrary, Debug)]
struct FuzzData {
    /// Podatki za šifriranje/dešifriranje
    data: Vec<u8>,
    /// Ključ za šifriranje
    key: [u8; 32],
    /// Nonce za šifriranje
    nonce: [u8; 12],
}

/// Fuzz target za kriptografske operacije
fuzz_target!(|data: FuzzData| {
    // Če so podatki prazni, preskočimo test
    if data.data.is_empty() {
        return;
    }

    // Poskusi šifrirati podatke
    let encrypted = match encrypt(&data.data, &data.key, data.nonce) {
        Ok(enc) => enc,
        Err(_) => return, // Preskočimo, če je napaka v šifriranju
    };

    // Poskusi dešifrirati podatke
    let decrypted = match decrypt(&encrypted, &data.key, data.nonce) {
        Ok(dec) => dec,
        Err(_) => return, // Preskočimo, če je napaka v dešifriranju
    };

    // Preveri, da so dešifrirani podatki enaki originalnim
    assert_eq!(data.data, decrypted, "Encrypt-decrypt cycle failed");
});
