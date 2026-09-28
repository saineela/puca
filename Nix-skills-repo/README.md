# Nix Skills Repository Example

This folder is a local example of the public GitHub repository format NIX PUCA reads in its Skills marketplace. It is not automatically trusted, executable, or connected to GitHub. Publish it only after reviewing its contents; the app installs declared files as a static package and does not execute them.

## Required layout

```text
Nix-skills-repo/
├── README.md
├── nix-skills.json
└── skills/
    └── web-search/
        ├── skill.json
        ├── README.md
        └── instructions.md
```

The repository root file **must** be named `nix-skills.json`. It declares the listing order and points to each skill manifest. The current importer reads the repository's default branch and supports public GitHub repository root URLs such as `https://github.com/owner/repository`—not private repositories, release archives, branch URLs, or raw-file URLs.

## Repository manifest: `nix-skills.json`

```json
{
  "schema_version": 1,
  "kind": "nix-skills-repository",
  "name": "Example NIX Skills",
  "description": "A short description of this collection.",
  "publisher": "Your display name or organization",
  "skills": [
    { "manifest": "skills/web-search/skill.json" }
  ]
}
```

A repository can list up to 24 skills. Each path must match `skills/<lowercase-slug>/skill.json`; the `id` in that manifest must match the folder slug.

## Skill manifest: `skills/<id>/skill.json`

```json
{
  "schema_version": 1,
  "kind": "nix-skill",
  "id": "web-search",
  "name": "Web Search",
  "version": "0.1.0",
  "description": "A clear description of what the skill is intended to do.",
  "category": "Research",
  "publisher": "Your display name or organization",
  "license": "MIT",
  "icon": "⌕",
  "tags": ["search", "research"],
  "permissions": ["Requests access to public web pages when execution is supported."],
  "files": ["README.md", "instructions.md"]
}
```

Fields `schema_version`, `kind`, `id`, `name`, `version`, `description`, and `license` are required. `publisher` defaults to the repository publisher; `category` defaults to `Tools`; `icon`, `tags`, `permissions`, and `files` are optional. `icon` is a short text glyph (not an image URL). File paths are relative to the skill's folder. An optional `entrypoint` must name a path already in `files`; it is descriptive metadata only today.

## Package and safety limits

- Up to 24 skills per repository, 25 declared files per skill, 256 KiB per file, and 1 MiB total per installed package.
- Manifests are limited to 64 KiB; files must use safe relative paths (no absolute paths, `..`, symlinks, archives, or file/directory conflicts).
- Only HTTPS GitHub repository metadata and `raw.githubusercontent.com` files are fetched; redirects are rejected.
- NIX copies only explicitly listed files under its local skills package directory. It does not unpack archives, import modules, run scripts, grant permissions, or enable a skill. Installed means “static package saved,” not “runnable.”
- Community packages are unreviewed publisher content, not verified or endorsed by NIX. Review each repository and its stated license/permissions before installing.

Keep metadata accurate, use a license you are authorized to publish under, and never include secrets, personal data, model weights, or unrelated files. This format is an MVP and may evolve; increment `schema_version` for incompatible changes.
