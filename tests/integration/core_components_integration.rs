// Integration test za osnovne komponente TallyIO
// Ta test preverja interakcijo med več ključnimi moduli

// Uvozi potrebne module
use core::{Arena, Queue};

#[test]
fn integration_test_arena_and_queue() {
    // Ustvari areno za alokacijo pomnilnika
    let arena = Arena::new();

    // Ustvari vrsto za posredovanje sporočil
    let queue = Queue::new();

    // Alociraj nekaj vrednosti v areni
    let values = vec![
        arena.alloc(10),
        arena.alloc(20),
        arena.alloc(30),
        arena.alloc(40),
        arena.alloc(50),
    ];

    // Dodaj vrednosti v vrsto
    for &value in &values {
        queue.push(value);
    }

    // Preveri, da so vrednosti pravilno dodane
    assert_eq!(queue.len(), values.len());

    // Preveri, da so vrednosti pravilno vzete iz vrste (FIFO)
    for &expected in &values {
        let actual = queue.pop().unwrap();
        assert_eq!(actual, expected);
    }

    // Preveri, da je vrsta prazna
    assert!(queue.is_empty());
}

#[test]
fn integration_test_end_to_end() {
    // Nastavitev osnovnih komponent
    let arena = Arena::new();
    let queue = Queue::new();
    
    // Simuliraj MEV procesno verigo
    
    // 1. Korak: Alokacija podatkov za priložnosti
    let opportunity_ids = vec![
        arena.alloc("arbitrage_uniswap_sushiswap"),
        arena.alloc("liquidation_aave"),
        arena.alloc("sandwich_pancakeswap"),
        arena.alloc("frontrun_tx_01"),
        arena.alloc("backrun_tx_02"),
    ];
    
    // 2. Korak: Dodajanje priložnosti v vrsto za procesiranje
    for &id in &opportunity_ids {
        queue.push(id);
    }
    
    // 3. Korak: Simulacija procesiranja MEV priložnosti
    let mut processed_results = Vec::new();
    
    while !queue.is_empty() {
        // Izvleci naslednjo priložnost iz vrste
        let opportunity_id = queue.pop().unwrap();
        
        // Simuliraj evalvacijo priložnosti (v pravem sistemu bi bili tu kompleksni izračuni)
        let profit = match opportunity_id {
            id if id == opportunity_ids[0] => 1000, // arbitraža ima visok profit
            id if id == opportunity_ids[1] => 5000, // likvidacija ima najvišji profit
            id if id == opportunity_ids[2] => 800,  // sandwich trading
            id if id == opportunity_ids[3] => 300,  // frontrunning
            id if id == opportunity_ids[4] => 200,  // backrunning
            _ => 0,
        };
        
        // Shrani rezultat
        processed_results.push((opportunity_id, profit));
        
        // V realnem MEV sistemu bi tu imeli tudi:
        // - Izračun gas stroškov
        // - Preverjanje časovne kritičnosti
        // - Izogibanje MEV zaščitam
        // - Optimizacijo za nizko latenco
    }
    
    // 4. Korak: Validacija rezultatov
    assert_eq!(processed_results.len(), opportunity_ids.len());
    
    // Preveri, da je likvidacija najbolj dobičkonosna (najbolj pomembno za MEV)
    let most_profitable = processed_results.iter()
        .max_by_key(|&(_, profit)| profit)
        .unwrap();
    
    assert_eq!(most_profitable.0, opportunity_ids[1]); // likvidacija naj bi bila najbolj dobičkonosna
    assert_eq!(most_profitable.1, 5000);
    
    // Preveri, da so vsi rezultati ustrezni
    for &(id, profit) in &processed_results {
        assert!(profit > 0, "Priložnost ne sme imeti negativnega profita");
        assert!(opportunity_ids.contains(&id), "ID priložnosti mora biti veljaven");
    }
    
    // Preveri, da vrsta ni vsebovala nobenih drugih priložnosti
    assert!(queue.is_empty());
}
