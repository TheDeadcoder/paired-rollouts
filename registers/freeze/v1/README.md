# Pre-registration v1 freeze

`docs/PREREGISTRATION.md` at the commit whose message begins "Freeze pre-registration v1" is the frozen document. This directory holds its SHA-256 (`PREREGISTRATION.md.sha256`), the RFC 3161 request (`prereg.tsq`) and two independent timestamp tokens over that request (`prereg_freetsa.tsr` from https://freetsa.org/tsr and `prereg_digicert.tsr` from http://timestamp.digicert.com), the FreeTSA certificate chain used for verification (`freetsa_cacert.pem`), and `freeze.json` with the hash, the TSAs, the times they certified and the commit hash (filled in by the following commit, because a commit cannot record its own hash in a file it contains).

Verify (from the repository root, at or after the freeze commit):

```
shasum -a 256 docs/PREREGISTRATION.md
cat registers/freeze/v1/PREREGISTRATION.md.sha256
openssl ts -verify -data docs/PREREGISTRATION.md -in registers/freeze/v1/prereg_freetsa.tsr -CAfile registers/freeze/v1/freetsa_cacert.pem
openssl ts -reply -in registers/freeze/v1/prereg_freetsa.tsr -text | grep -E "Time stamp|Hash Algorithm|Message data" -A 2
openssl ts -reply -in registers/freeze/v1/prereg_digicert.tsr -text | grep -E "Time stamp|Hash Algorithm|Message data" -A 2
```

The first two lines must agree; the FreeTSA verification must print `Verification: OK`; both replies must show the same message digest as the request and a time stamp on the freeze date. The DigiCert token is verified against DigiCert's public timestamping roots, which are not vendored here; its reply text is sufficient to read the certified time.
