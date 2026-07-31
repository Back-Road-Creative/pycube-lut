# Security Policy

## Supported versions

Only the latest released tag receives security fixes. Pin a released `v*` tag; `main`
is unstable.

## Threat model in one paragraph

`load_cube()` parses **untrusted input** — a `.cube` is a text file and a HALD CLUT is
a PNG, and both are routinely downloaded from strangers. The parser is pure Python and
NumPy: it does no `eval`, runs no code from the file, opens no network connection, and
writes nothing to disk. A malformed file should always raise `CubeError`, never crash
the interpreter, hang, or produce a silently truncated table. PNG decoding is delegated
to Pillow, so Pillow's own advisories apply to that path — keep it current.

Anything that violates that paragraph is a security bug. In particular:

- an uncaught exception type (anything that is not `CubeError`) from `load_cube()` on a
  malformed or hostile file;
- unbounded memory or time from a small input (for example a size directive that makes
  the reader allocate a table the file cannot possibly contain);
- a path that reads or writes anything the caller did not name.

## Reporting a vulnerability

Please report suspected vulnerabilities privately. Do **not** open a public issue.

- Preferred: open a private advisory via GitHub's **Security → Report a vulnerability**
  tab on this repository.
- Fallback (if that tab is unavailable to you): email the maintainer address published
  in this project's package metadata (the `authors` entry in `pyproject.toml`, also
  shown on the PyPI page), with `pycube-lut security` in the subject.

Please include the affected version or commit, the impact, reproduction steps or a
proof of concept (the smallest LUT file that triggers it), and any suggested fix.

## What to expect

- Acknowledgement within 5 business days.
- Initial assessment and severity triage within 10 business days.
- Coordinated disclosure: we will agree a timeline with you before any public write-up,
  and credit reporters who want it.

## Out of scope

- Vulnerabilities in NumPy or Pillow themselves — report those upstream.
- A LUT that produces ugly or unexpected colour. That is a taste or colour-management
  question, not a security issue.
