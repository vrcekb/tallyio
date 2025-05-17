//! Fuzz test za Queue implementacijo
//! 
//! Ta fuzz test preverja robustnost Queue implementacije proti
//! naključnim sekvencam push in pop operacij, kar je ključno za
//! zanesljivo delovanje v MEV okolju.

#![no_main]
use libfuzzer_sys::fuzz_target;
use core::Queue;
use arbitrary::Arbitrary;

/// Definiraj operacije, ki jih lahko izvajamo na vrsti
#[derive(Arbitrary, Debug)]
enum QueueOperation {
    /// Dodaj vrednost v vrsto
    Push(i32),
    /// Vzemi vrednost iz vrste
    Pop,
}

/// Fuzz target, ki izvaja zaporedje operacij na vrsti
fuzz_target!(|operations: Vec<QueueOperation>| {
    // Ustvari vrsto
    let queue = Queue::new();
    
    // Ustvari referenčno implementacijo (standardni Vec) za primerjavo
    let mut reference = Vec::new();
    
    // Izvedi operacije
    for op in operations {
        match op {
            QueueOperation::Push(value) => {
                // Dodaj v vrsto in v referenco
                queue.push(value);
                reference.push(value);
            }
            QueueOperation::Pop => {
                // Vzemi iz vrste in iz reference
                let queue_result = queue.pop();
                let reference_result = reference.first().map(|&v| v);
                if !reference.is_empty() {
                    reference.remove(0);
                }
                
                // Preveri, da se rezultati ujemajo
                assert_eq!(queue_result, reference_result, 
                    "Queue pop returned {:?}, but reference returned {:?}", 
                    queue_result, reference_result);
            }
        }
    }
});
