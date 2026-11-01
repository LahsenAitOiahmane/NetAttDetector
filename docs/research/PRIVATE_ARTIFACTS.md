# Registered private research-model transfer

Registered 1 November 2026 before exporting a model. Public GitHub jobs currently discard
private fitted models with the runner. Retaining a frozen model for Windows verification and
later independent evaluation requires a private transfer, not a public weights release.

Generate an X25519 recipient key locally. Keep the private32-byte key in an ignored local
directory with owner-only permissions; commit only the public recipient key/fingerprint.
The private key never enters GitHub, repository secrets, logs or artifacts. Its backup/access
remains the owner's responsibility. This workflow does not authorize publishing plaintext models.

Encrypt an explicitly bounded JSON payload containing only the fitted model bytes, public
freeze metadata and **synthetic** portability inputs/expected scores. No real dataset rows,
labels, endpoint/UID values or credentials enter the payload. Use an ephemeral X25519 exchange,
HKDF-SHA256 with a fresh salt and context binding, and AES-256-GCM with a fresh nonce and
authenticated header. Use maintained cryptography primitives, not custom cryptographic math.
The versioned envelope rejects unsupported algorithms, malformed fields, wrong recipients,
oversized/truncated files and failed authentication. No private key/model data is printed.

Ciphertext may be retained as a GitHub artifact; plaintext stays private. Since anyone with
the public key could encrypt another payload, decryption alone does **not** authenticate its
sender. Before any pickle/joblib load, compare the recovered model SHA-256 to a separately
trusted committed diagnostic freeze from the owner's workflow. Do not trust a hash found only
inside an arbitrary encrypted payload. Decryption writes to a fresh ignored destination and
never overwrites models or keys.

Verify portability locally using synthetic feature vectors only, with the same input contract,
recorded dependency versions and fixed prediction thread. Report maximum score difference
and threshold-decision differences, not synthetic attack accuracy. This establishes only the
tested probe behavior, not all-platform/model generalization or real-network performance.

The project owner requested complete offline research and selected MIT for code. This protocol
retains the user's research model privately; it does not change dataset terms, candidate label
promotion, public-model permissions or the explicitly deferred pilot. No live traffic is captured.
