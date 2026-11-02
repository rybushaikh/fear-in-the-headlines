# Security

**Reporting.** Please report a vulnerability or an exposed secret privately through GitHub's "Report a vulnerability" option on the Security tab. Please don't open a public issue.

**How this repo stays safe**
- API keys live only in a local `.env`, which is git-ignored. `.env.example` holds empty names only.
- `.gitignore` is default-deny: a file can be committed only if a rule explicitly allows it.
- Every push and pull request runs a gitleaks secret scan.
- No headline text, labels, per-headline scores or raw API responses are stored here.

If you run the collectors, use your own keys and respect each provider's terms and rate limits.
