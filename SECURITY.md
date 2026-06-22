# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in `ace-rs`, please report it
**privately**. Do **not** open a public GitHub issue, pull request, or
discussion for security-sensitive reports.

Instead, either:

- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" under the repository's **Security** tab), or
- Email **thehounddog@protonmail.com** with the details.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof of concept.
- The affected version(s) or commit.

You can expect an initial acknowledgement within **5 business days**. We will
keep you informed of progress toward a fix and coordinate a disclosure timeline
with you.

## Scope

`ace-rs` is a low-level primitives library. Of particular interest are:

- Incorrect results from a native (intrinsic) path versus the scalar oracle
  that could lead to silent data corruption.
- Unsoundness in any `unsafe` block (e.g. feature detection gating an
  intrinsic that the CPU does not actually support).

## Supported versions

This project is pre-1.0; security fixes are applied to the latest released
version on a best-effort basis.
