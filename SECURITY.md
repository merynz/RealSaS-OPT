# Security Policy

RealSaS is proprietary research/product software. Please do not publish a suspected security issue, credential, private artifact locator, model secret, or exploitable repository detail in a public issue or discussion.

## Reporting

Report security-sensitive findings privately to the repository owner through the GitHub account/contact channel associated with this repository. Include:

- affected commit/path/component;
- reproduction steps;
- impact and prerequisites;
- whether credentials, private artifacts or user data may be involved;
- a minimal proof of concept when safe to provide.

Do not include production credentials, private tokens, personal data, or large proprietary artifacts in the report unless specifically requested through a secure channel.

## Supported scope

Security fixes target the current promoted mainline and active authorized research apparatus. Historical branches, sealed experiments and provenance records may be preserved for scientific reproducibility even when they are not supported as executable software.

## Secrets and artifacts

- never commit API tokens, passwords, private keys or cloud credentials;
- `.env*` local overrides are ignored except for a deliberately committed `.env.example`;
- large scientific artifacts should be content-hash bound and stored in their authorized external artifact location rather than committed merely for convenience;
- a checkpoint/result hash is evidence identity, not permission to redistribute the bytes.

## Dependency issues

Current developer/CI dependency profiles are under `requirements/`. A dependency alert does not retroactively alter sealed experiment environments; scientific replay environments must remain provenance-preserving and be remediated through a separately documented compatibility/replay decision when necessary.
