# Security

Do not post credentials, private data, or exploitable details in public issues.
Use GitHub's private vulnerability reporting when it is available.

Install dependencies from the repository requirements or lockfile in an
isolated environment. Keep credentials in environment variables or the hosting
provider's secret store. Never commit local data, access tokens, or credentials.

Current Python entry points use restricted checkpoint loading with a small
NumPy allowlist for legacy RNG state. This is not a sandbox against resource
exhaustion: load only checkpoints from a trusted source, and verify published
hashes. Deployment checks the model hash before loading it.

The historical experiment notebooks pin earlier code revisions and validation
records. Reproducing those experiments requires their original trusted assets;
updating this repository does not change their pinned code or validate a new
Colab training environment.

The FastAPI upload endpoint bounds the request before multipart parsing and
keeps one file in memory. Do not submit identifying or patient images to a
public demo. Hosting infrastructure can retain ordinary request metadata.
