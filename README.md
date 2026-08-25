# HE-VeriTrust

**Verifiable-attestation federated intrusion detection under homomorphic encryption**

Federated learning for IoT intrusion detection has to survive two adversaries at
once: a curious server that can invert model updates back into private traffic,
and a Byzantine minority that poisons the global model. The two standard
remedies conflict — Byzantine-robust aggregators must *read* the plaintext
updates that encryption is meant to *hide*.

The usual escape is to let each client send a small plaintext "attestation"
describing its own update, and to gate aggregation on that. **That escape does
not work.** An adversary able to craft a poisoned update is equally able to type
benign numbers next to it, so the defence ends up scoring evidence supplied by
the entity it is policing.

HE-VeriTrust removes the client from the evidence path. The robustness signal is
**measured from the ciphertext** the client actually submitted, using the
additive homomorphism itself:

```
Π_j E(q_j)^{v_j}  =  E( ⟨q, v⟩ )        for a public probe vector v
```

`k` such probes give a `k`-dimensional sketch of every update, computed without
any client cooperation and without revealing the update. A client cannot make
its attestation disagree with its update, because the attestation *is* a
measurement of the update.

 
---

 

---

## Security properties,  

**1. Attestation cannot be forged.** Trust features (`proj_ref`, `norm_ratio`,
`peer_agreement`) are derived by the decryption authority from the submitted
ciphertexts. Probes are seeded by `H(round ‖ transcript_root ‖ authority_nonce)`,
so a client must commit to its update *before* the measured subspace exists —
commit-then-reveal, refreshed every round.

**2. The authority is a verifier, not an oracle.** It verifies every client's
Ed25519 signature, derives the probes itself, **recomputes** every ciphertext it
decrypts, enforces a participation floor and a per-client weight cap, and opens
exactly one aggregate per round. This is what closes the singleton and
difference attacks — note a `(t, n)` threshold committee alone does *not* close
the difference attack, because each member sees a perfectly legitimate
aggregate.

**3. Encryption does not dictate the model size.** BatchCrypt-style slot packing
carries 50 quantised coordinates per 2048-bit plaintext, with the no-carry bound
`Λ·(2^v − 1) < 2^slot_bits` asserted before every opening rather than assumed.

**4. Leakage is measured against the optimal attack.**  

---

## Install

```bash
python -m venv .venv && .venv\Scripts\activate     
pip install -r requirements.txt
```

Python ≥ 3.9. `gmpy2` matters: `phe` auto-detects it and it gives roughly a 10×
speed-up on the modular exponentiation that dominates Paillier.

## Data

CIC-IoT-2023 and Edge-IIoTset are third-party and not redistributed. Point the
config at your local copy:

```yaml
# configs/cic_iot.yaml
data:
  csv_root: "/path/to/CIC_IoT_Attack_2023/CSV"   # the per-attack-type tree
```

The loader reads the **per-attack-type directory tree**, not the pre-merged
sample files. That choice is load-bearing: the merged files are a uniform
subsample, so they shrink the rare families to ~1.5 k (Web) and ~0.8 k
(BruteForce) flows against 50 k per flood family, which makes those classes
statistically unlearnable and depresses macro-F1 for reasons that have nothing
to do with the detector. Reading the full tree recovers them to ~24.8 k and
~13.1 k, taking the imbalance ratio from ~65:1 to ~3:1.

Feature selection and scaler fitting both happen **after** the train/val/test
split and on the training split only.

 

Useful flags: `--scenarios`, `--seeds`, `--rounds`, `--attack {sign_flip,ipm,alie,min_max,min_sum}`,
`--jobs N` (Paillier workers; set it when running several seeds in parallel so
the process pools do not oversubscribe the CPU), `--quick` (smoke test).

## Layout

```
configs/           one YAML per dataset - the single source of truth
src/crypto/        packing, Paillier roles, identity/transcripts, sketch, authority
src/federated/     client, server, coordinated attacks, plaintext baselines
src/trust/         sketch-derived features, ANFIS + Mamdani engines, Zero-Trust policy
src/data/          per-attack-type CIC loader, Dirichlet partitioner
 

 
