Lockfiles must be consistent with package metadata. After any change to `pixi.toml`, run `pixi lock`.

Everything runs in a pixi environment. Any command requiring the installed python environment (like `pytest`) must be prefixed with `pixi run` (e.g. `pixi run pytest` or `pixi run python`); local editing/inspection tools do not.

Code formatting must align with our standards. Run `pixi run lint` before `git commit`s to ensure this.

## Code Style
- Code should be secure, performant, elegant, robust and maintainable.
- No overly conservative default values like `dict.get(key, default)` as this may hide missed wiring
- No blanked `from __future__ import annotations` if there are no forward/self references
- No hacky workarounds to ensure backward compatibility if a meaningful code change has been implemented. If critical code hinges on compatibility, ask back for advice
- No `git stash && ... ` to verify 'what might have passed before'. All test should pass always.
