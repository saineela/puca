<p align="center"><a href="https://github.com/saineela/puca"><img src="https://res.cloudinary.com/dh5uxc6ql/image/upload/v1790917615/93d18a47-5f3a-46e1-ad71-705b2680442f_anp8f1.png" alt="NIX PUCA" width="320"></a></p>

<p align="center"><a href="https://github.com/saineela/puca/stargazers">☆ Star NIX PUCA on GitHub</a></p>

# Create a skill for NIX PUCA

This guide shows you how to package a skill so NIX PUCA can find it, describe it clearly, and (when a supported runtime is present) understand its tools. You do not need to write executable code to make a valid **static skill**.

> **Important:** Installing a skill saves its files; it does not run them. A runnable community skill needs a supported NIX worker, setup fields, tool schemas, and a separate explicit trust step. Trusted code runs with the NIX account's normal file and network access; it is not OS-sandboxed. Never upload secrets.

## 1. Choose where the skill comes from

- **ZIP upload:** Make one ZIP for one skill. Put `skill.json` at the ZIP root and include every file named in its `files` list. ZIP imports cannot auto-update; upload a new ZIP when you publish a new version.
- **Public GitHub repository:** Use this layout to publish one or more skills. NIX previews the repository's default branch; users can opt into checking for updates when they open the Skills page.

```text
my-nix-skills/
├── nix-skills.json             # required for GitHub repositories
└── skills/
    └── friendly-greeter/
        ├── skill.json          # required
        ├── README.md           # recommended: explain what it does
        ├── instructions.md     # optional: plain-language guidance
        └── logo.svg            # optional: PNG/JPEG/WebP/SVG logo
```

For a ZIP upload, place the contents of `skills/friendly-greeter/` directly at the ZIP root. Do not include a parent folder around the files.

## 2. Write `skill.json`

The manifest is the skill's identity card. NIX uses `kind`, `schema_version`, `id`, `name`, `description`, and `files` to recognize the package. The ID must be a short lowercase slug and match its GitHub folder name.

```json
{
  "schema_version": 1,
  "kind": "nix-skill",
  "id": "friendly-greeter",
  "name": "Friendly Greeter",
  "version": "1.0.0",
  "description": "Writes a short, friendly greeting in the style you choose.",
  "category": "Writing",
  "publisher": "Your name or organization",
  "license": "MIT",
  "icon": "✦",
  "logo": "logo.svg",
  "tags": ["writing", "greetings"],
  "permissions": [],
  "files": ["README.md", "instructions.md", "logo.svg"]
}
```

Required fields are `schema_version` (currently `1`), `kind` (`nix-skill`), `id`, `name`, `version`, `description`, `license`, and `files`. `publisher` defaults to the repository publisher, and `category` defaults to `Tools`. `icon` is an optional short text fallback. `logo` is optional; it must be a relative path to a file also listed in `files`. Use SVG, PNG, JPEG, or WebP, up to 256 KiB. SVG logos must be static artwork (no scripts, animation, embedded images, or external links).

Every path in `files` is relative to the skill folder (or the ZIP root). NIX installs only files explicitly listed there. Keep paths simple; do not use absolute paths, `..`, symlinks, or ZIP entries that are not declared in the manifest.

## 3. Help NIX understand the skill

Write the description for a person who has never seen the package: what it does, when it is useful, and its main limitation. Add a `README.md` with setup steps and examples. Use clear, ordinary language. Explain what inputs the skill expects and what a successful result looks like.

A static package can provide information and instructions, but that alone does **not** make NIX execute it or call an external service. To describe an executable skill for the supported NIX worker, the manifest needs `runtime`, `setup_fields`, `tools`, and `triggers`. The runtime protocol is `nix-skill-jsonl-v1`; every tool needs a bounded JSON `input_schema` and `result_schema`. See [`skills/ring-light/skill.json`](skills/ring-light/skill.json) for a complete example. A manifest declaration is not an approval: NIX still requires package setup and the user to review and trust that exact package digest before it can run.

## 4. Publish or upload

**GitHub:** Keep `nix-skills.json` in the public repository root. It lists skill manifests in the form:

```json
{
  "schema_version": 1,
  "kind": "nix-skills-repository",
  "name": "My NIX skills",
  "description": "Tools and helpers I made for NIX PUCA.",
  "publisher": "Your name or organization",
  "skills": [{ "manifest": "skills/friendly-greeter/skill.json" }]
}
```

Paste the public repository root URL (`https://github.com/owner/repository`) into Skills and preview it. NIX does not accept private repositories, branch URLs, or raw-file links.

**ZIP:** Select a `.zip` file in the Skills upload area. The archive must contain one `skill.json` at its root and the declared package files. Keep the archive under 2 MiB; each skill file is limited to 256 KiB and the installed package to 1 MiB. NIX checks archive paths and contents and installs only declared files.

## 5. Updating and safety

GitHub skill updates are opt-in. Turn **Auto-updates** on for a GitHub skill to let NIX check its repository when the Skills page opens, at most once every six hours. Bump the skill's `version` whenever you publish an update. If an opted-in skill's version changes, NIX installs the new static package and clears its setup and previous trust; review and configure it again before using it. Turning auto-updates off stops these checks. ZIP-uploaded skills always have auto-updates off and do not have a remote update source.

Marketplace size limits: 24 skills per GitHub repository, 25 files per skill, 64 KiB per manifest, 256 KiB per file, and 1 MiB per installed package. Never include private data, API keys, passwords, unrelated files, or material you do not have permission to distribute. State permissions honestly and include a license you are allowed to use.

## Example package

The [`skills/ring-light/`](skills/ring-light/) folder demonstrates a worker-enabled manifest, explicit permissions, setup fields, tool input/result schemas, trigger phrases, and a plain-language README. It controls real hardware only after a user supplies device configuration and explicitly reviews and trusts the installed package.