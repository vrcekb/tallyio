//! Ključi za šifriranje

use ring::rand::{SecureRandom, SystemRandom};
use std::fmt;
use zeroize::ZeroizeOnDrop;

/// Šifrirni ključ
#[derive(Clone, zeroize::Zeroize)]
pub struct Key(pub(crate) [u8; 32]);

impl ZeroizeOnDrop for Key {}

impl Key {
    /// Vrne referenco na surove bajte ključa
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key([...])")
    }
}

impl Key {
    /// Ustvari nov ključ iz obstoječih podatkov
    #[must_use]
    pub const fn from_slice(data: &[u8]) -> Option<Self> {
        if data.len() != 32 {
            return None;
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(data);
        Some(Self(key))
    }

    /// Generira nov naključen ključ
    /// # Errors
    /// Vrne napako, če generiranje naključnih podatkov ne uspe.
    pub fn generate() -> Result<Self, ring::error::Unspecified> {
        let rng = SystemRandom::new();
        let mut key = [0u8; 32];
        rng.fill(&mut key)?;
        Ok(Self(key))
    }

    /// Vrne referenco na surove bajte ključa
    #[must_use]
    pub const fn raw_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl AsRef<[u8]> for Key {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let key = Key::generate().unwrap();
        assert_eq!(key.as_ref().len(), 32);
    }

    #[test]
    fn test_key_from_slice() {
        let data = [42u8; 32];
        let key = Key::from_slice(&data).unwrap();
        assert_eq!(key.as_ref(), &data);
    }

    #[test]
    fn test_key_from_invalid_slice() {
        let data = [42u8; 16];
        assert!(Key::from_slice(&data).is_none());
    }

    #[test]
    fn test_key_debug() {
        let key = Key::generate().unwrap();
        assert_eq!(format!("{key:?}"), "Key([...])");
    }

    #[test]
    fn test_key_raw_bytes() {
        // Test za raw_bytes metodo
        let data = [42u8; 32];
        let key = Key::from_slice(&data).unwrap();

        // Preverimo, da raw_bytes vrne pravilne podatke
        assert_eq!(key.raw_bytes(), &data);

        // Preverimo, da je rezultat enak kot pri as_ref
        assert_eq!(key.raw_bytes(), key.as_ref());
    }
}
