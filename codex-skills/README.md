# Personal Codex skills

Snapshot of personal skills from `~/.codex/skills/` and `~/.agents/skills/`. Each folder in `skills/` includes its supporting files. Built-in system skills and plugin-managed skills are excluded.

## Install on another laptop

After cloning this repository, copy the folders inside `skills/` into your home directory's `.agents/skills/` folder. Preserve each skill folder and its contents.

On macOS or Linux, run from the repository root:

```sh
mkdir -p "$HOME/.agents/skills"
cp -R codex-skills/skills/. "$HOME/.agents/skills/"
```

The copy command replaces files with matching names; back up any locally modified versions first. On Windows, copy the skill folders into `%USERPROFILE%\.agents\skills`.

Restart Codex if the skills do not appear. Install or connect any plugins and external tools required by individual skills separately. Machine-specific absolute paths inside skills may need updating on the new laptop.

## Included skills

- `author-voice-profile`
- `blindspot`
- `paper-voice-review`
- `referee2`
- `cavecrew`
- `caveman`
- `caveman-commit`
- `caveman-compress`
- `caveman-discover`
- `caveman-evidence-review`
- `caveman-explore`
- `caveman-help`
- `caveman-learn`
- `caveman-manage`
- `caveman-optimize`
- `caveman-review`
- `caveman-setup`
- `caveman-stats`
- `investigate-first`
- `lean-build`
- `megacave`
- `migration`
- `safe-refactor`
- `surgical-patch`
- `ultracave`
- `verify-and-stop`
