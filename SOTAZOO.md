# SotaZoo Penpot Fork

This file records SotaZoo-specific context for the public fork at `SotaZoo/penpot`.
It is intentionally separate from upstream Penpot documentation.

## Branches

- `develop` - default fork branch for general repo discovery.
- `mcp-stale-plugin-recovery` - upstream-style implementation branch for the MCP stale-session recovery fix.
- `mcp-stale-plugin-recovery-2.16.1` - release-line backport branch used by the live MBP-hosted Penpot server.

## Live Environment

The current live SotaZoo self-hosted Penpot server uses the 2.16.1 backport MCP image:

```text
sotazoo/penpot-mcp:2.16.1-sz1-4053cae
```

The image tag records the code commit used for the live MCP build. Documentation-only commits may appear after that commit on this repo.

Working rule: when working against the live SotaZoo self-hosted Penpot server, assume the custom Penpot MCP code comes from `mcp-stale-plugin-recovery-2.16.1` unless current repo or server evidence shows otherwise.

## Fix Context

The SotaZoo fork carries an MCP stale-session recovery fix so agent-driven Penpot work can reconnect after the plugin session becomes stale.

The upstream-style branch exists for future upstream/contribution alignment. The 2.16.1 branch exists because the live server is pinned to the Penpot 2.16.1 release line.

Known separate issue: exporting the full root/page can timeout in some wireflow canvases. Selection export worked in the verified scorekeeper workflow. Treat full-canvas export reliability as a separate follow-up, not as evidence that the stale-session recovery fix failed.
