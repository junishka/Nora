# Security policy

Nora's core claim is that the frontier model never sees the raw data.
Vulnerability reports that affect this claim, or any of the layers that
carry it (the MCP tool interface, the macOS sandbox, the disclosure
control sanitizer, credential storage), take priority.

## Reporting a vulnerability

Preferred channel: GitHub's private vulnerability reporting. Open the
**Security** tab on this repo and choose **Report a vulnerability**.
This routes a private advisory to the maintainer; the issue stays out
of public view until disclosure is coordinated.

Alternatively, email the maintainer at jbyun@iese.edu. Please do not
file public issues for security reports.

Please include:

- A minimal reproduction (steps, environment, Nora version).
- Which layer the issue affects (tool interface, sandbox, sanitizer,
  packaging, auth).
- Whether the issue is exploitable against the documented threat model
  (a curious or adversarial frontier model trying to exfiltrate data),
  or only against a researcher who actively undermines the local
  safeguards.

## What is in scope

- Anything that lets the model observe raw data values, free-text
  values, or low-count cells past the sanitizer.
- Anything that lets the model issue arbitrary filesystem, network, or
  shell operations from inside the sandbox.
- Auth or keyring handling that exposes API keys to the model or
  writes them to disk in cleartext.
- Packaging issues (signing, notarization, supply chain) that affect
  the integrity of a downloaded `.dmg`.

## What is out of scope

- Running Nora outside macOS. The sandbox boundary is macOS-specific by
  design; the package refuses script execution on other platforms.
- A researcher willingly pasting their own raw data into a chat
  message. The model can read what the researcher types; this is by
  design.
- Vulnerabilities in upstream dependencies that do not affect Nora's
  privacy invariants.

## Disclosure timeline

Best-effort acknowledgment within seven days. Fixes are prioritized by
severity. Coordinated disclosure is preferred so a fix can ship before
public detail.

## Supported versions

The current beta line (`0.11.x`) receives security fixes. Earlier
beta minors (`0.10.x` and below) and pre-beta versions (`0.0.x`) are
no longer supported. After the first stable release (`1.0.0`), this
section will document the support window for the previous minor.
