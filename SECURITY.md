# Security policy

Decision Model Guard is a security layer, so a bypass is a vulnerability: a way to steer a guarded decision that the
documented guarantees say should not work, a crash or resource exhaustion triggered by content, or a leak of scene text
through logs or the API.

## Reporting

Please report vulnerabilities privately through **"Report a vulnerability"** on the
[Security tab](https://github.com/patronus-protect/decision-model-guard/security) of this repository, not in a public
issue. Include the scene, the question, the options, the model and the configuration that reproduce the problem.

We aim to confirm a report within five working days and to agree on a disclosure date with you.

## Scope

Known limits are documented in the README (section "Limits") and are not vulnerabilities by themselves: fabricated
facts that argue for a verdict without naming it, claims split across distant sentences, claims the claim detector
misses, and false positives that the cautious rules cannot repair. New ways around these limits that you can measure are
still very welcome as regular issues.

## Supported versions

Security fixes go into the latest release.
