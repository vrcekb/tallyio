fn main() {
    println!("TallyIO CLI zagnan.");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_main() {
        // Test za main funkcijo
        // Ker main funkcija samo izpiše sporočilo, je dovolj, da jo pokličemo
        // in preverimo, da se ne sesuje
        assert!(std::panic::catch_unwind(|| {
            super::main();
        })
        .is_ok());
    }
}
