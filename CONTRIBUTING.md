# Sodelovanje pri razvoju TallyIO

Hvala, da razmišljate o prispevanju k TallyIO! S tem vodičem vam bomo pomagali, da boste lahko enostavno prispevali k projektu.

## Kako lahko prispevam?

1. **Poročanje napak**
   - Če najdete napako, preverite, ali še ni navedena med [težavami](https://github.com/vrcekb/tallyio/issues).
   - Če težave še ni, jo ustrezno opišite in dodajte korake za ponovitev.

2. **Predlaganje izboljšav**
   - Imate idejo za izboljšavo? Odprite novo težavo in jo označite z oznako "enhancement".

3. **Prispevanje kodi**
   - Ustvarite fork projekta
   - Ustvarite vejo za svojo spremembo (`git checkout -b moja-izboljsava`)
   - Zapišite teste za svojo spremembo
   - Zagotovite, da vsi testi uspešno tečejo
   - Pošljite Pull Request

## Zahteve za kodo

- Sledite [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/)
- Vse funkcije morajo biti dokumentirane
- Vse spremembe morajo vsebovate teste
- Koda mora biti formatirana z `cargo fmt`
- Koda mora preiti vse Clippy preveritve

## Postopek za pošiljanje sprememb

1. Posodobite svoj fork z najnovejšimi spremembami:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Ustvarite novo vejo za svojo spremembo:
   ```bash
   git checkout -b opis-spremembe
   ```

3. Naredite spremembe in jih zabeležite:
   ```bash
   git add .
   git commit -m "Kratek opis spremembe"
   ```

4. Potisnite spremembe v vaš fork:
   ```bash
   git push origin opis-spremembe
   ```

5. Odprite Pull Request na GitHubu.

## Razvijalsko okolje

Za lokalni razvoj priporočamo uporabo sledečih orodij:

- Najnovejša stabilna različica Rust-a
- `rustup` za upravljanje različic Rust-a
- `rustfmt` za oblikovanje kode
- `clippy` za statično analizo kode
- `cargo-udeps` za odkrivanje neuporabljenih odvisnosti
- `cargo-audit` za preverjanje varnostnih ranljivosti

## Testiranje

Pred oddajo sprememb se prepričajte, da vsi testi uspešno tečejo:

```bash
cargo test --all
```

## Pravila za commit sporočila

- Uporabljajte jasna in jedrnata sporočila
- Za referenco uporabite številko težave (#123)
- Za večje spremembe poskrbite za ustrezen opis sprememb

## Licenca

Z oddajo kode soglašate, da bo vaš prispevek licenciran pod MIT licenco.

## Zahvala

Hvala, da ste si vzeli čas za prispevanje k TallyIO! Vaš prispevek je zelo cenjen.
