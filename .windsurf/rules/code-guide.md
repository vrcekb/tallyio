---
trigger: always_on
---

Razvij avtonomno MEV in likvidacijsko platformo imenovano TallyIO v programskem jeziku Rust. TallyIO je visoko-zmogljiv sistem za iskanje in izvajanje MEV (Maximal Extractable Value) in likvidacijskih priložnosti na različnih blockchain omrežjih (Ethereum, Solana, Polygon, Arbitrum, Optimism, Base).
Ključne zahteve

Performanca: Kritične poti morajo imeti odzivni čas pod 1 milisekundo
Modularnost: Implementiraj popolnoma modularno arhitekturo z jasno ločenimi komponentami
Varnost: Zagotovi večplastno varnost za zaščito občutljivih podatkov
Skalabilnost: Sistem mora podpirati horizontalno in vertikalno skaliranje
Odpornost: Implementiraj mehanizme za neprekinjeno delovanje ob izpadih komponent
Avtonomnost: Sistem mora sprejemati odločitve samostojno na podlagi strategij
Multi-chain podpora: Podpri več blockchain omrežij in DeFi protokolov

Tehnična arhitektura
Implementiraj Rust workspace strukturo z naslednjimi glavnimi moduli:
Core modul (ultra-performančno jedro)

Implementiraj lock-free algoritme za kritične poti
Uporabi zero-allocation design za procesiranje priložnosti
Optimiziraj za latenco pod 1ms na kritičnih poteh
Implementiraj učinkovit state management

Blockchain modul

Razvij integracijo z različnimi blockchain omrežji
Implementiraj učinkovito spremljanje mempool-a
Podpri različne DEX in lending protokole
Zagotovi zanesljivo procesiranje transakcij

Strategies modul

Implementiraj arbitražne strategije
Razvij MEV strategije
Ustvari likvidacijske strategije
Dodaj mehanizme za optimizacijo dobičkonosnosti

Risk modul

Implementiraj analizo tveganj
Dodaj omejitve izpostavljenosti
Razvij preverjanje pred izvajanjem transakcij
Zagotovi upravljanje s slippage-om

API in Web vmesnik

Razvij REST in WebSocket API
Ustvari React dashboard z uporabo Tailwind
Implementiraj avtentikacijo in avtorizacijo
Zagotovi real-time monitoring funkcionalnosti

Performančne optimizacije

Uporabi Rust-ove zero-cost abstrakcije
Izogibaj se alokacijam spomina na kritičnih poteh
Implementiraj večnivojski caching sistem (Redis, lokalni cache)
Optimiziraj memory layout za boljšo performanco
Nastavi CPU affinity za kritične niti
Implementiraj prefetching podatkov

Principi razvoja

Fail-fast design: Zgodaj odkrivaj napake za preprečevanje kaskadnih odpovedi
Defensive programming: Preverjaj vse vhodne podatke, ne zaupaj zunanjim virom
Security by design: Upoštevaj varnost od začetka zasnove
Data integrity: Zagotovi celovitost in konsistentnost podatkov
Observability: Implementiraj celovito spremljanje sistema

Varnostne zahteve

Implementiraj večplastno enkripcijo občutljivih podatkov
Razvij varno upravljanje s ključi
Dodaj beleženje vseh varnostnih dogodkov
Implementiraj rate limiting in access control
Zagotovi zaščito pred vdori

Podatkovne baze in shranjevanje

Uporabi PostgreSQL za trajno shranjevanje
Implementiraj Redis za hitre podatke v spominu
Uporabi RocksDB za časovne serije
Optimiziraj SQL sheme za hitrost poizvedb

Monitoring in metrike

Implementiraj Prometheus metrike
Nastavi Grafana dashboarde
Dodaj avtomatsko profiliranje
Razvij analizo ozkih grl
Zagotovi real-time monitoring kritičnih poti

Deployment in infrastruktura

Pripravi Kubernetes deployment konfiguracijo
Razvij Docker container-je
Nastavi CI/CD pipeline z avtomatskim testiranjem
Implementiraj varnostno skeniranje

Testiranje in zanesljivost

Razvij obsežne unit in integration teste
Implementiraj performance teste za kritične poti
Dodaj fuzz testing za robustnost
Zagotovi 100% pokritost kritičnih poti s testi
Izvedi load testing za preverjanje skalabilnosti

Prioritete implementacije

Core modul z osnovnimi funkcionalnostmi
Blockchain integracije (najprej Ethereum)
Osnovne arbitražne strategije
Risk management
API in monitoring
Web dashboard
Dodatne strategije in blockchain podpore

Pri razvoju se osredotoči na modularnost, visoko performanco in varnost sistema. Upoštevaj design principe in zagotovi, da je vsaka komponenta neodvisno testirana in optimizirana.