# Contributing to Personal AI

Thanks for your interest in improving Personal AI. This guide covers how to set
up, make changes, and submit them.

## Project principles

Keep these in mind, they shape every change:

- **Zero runtime dependencies.** The core runs on the Python standard library
  only. Optional features (voice, encryption) may use extra packages, but they
  must degrade gracefully when those packages are missing.
- **Local-first and private.** Data stays on the user's machine. Do not add
  code that sends data anywhere without an explicit, configured integration.
- **Offline-capable.** Features should work offline, or fail gracefully with a
  clear message when the internet is required.
- **Safe by default.** Tools that run code, run shell commands, or modify the
  assistant must stay behind their config flags and keep their guardrails.

## Development setup

You need Python 3.9 or newer. No packages to install for the core.

```bash
git clone https://github.com/cmatt11/personal-ai.git
cd personal-ai
python run.py --doctor      # check your environment
```

For a local model, install [Ollama](https://ollama.com) and pull the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Running tests

All tests run offline and must pass before you open a pull request:

```bash
python -m unittest discover -s tests -v
python -m compileall -q personal_ai     # verify everything compiles
```

Add tests for any new behavior. Put them in `tests/test_core.py`. Tests should
not require a network connection or a real model; use `MockBackend` for the LLM
and `HashingEmbeddings` for embeddings.

## Code style

- Follow standard PEP 8. Keep functions focused and readable.
- Write clear docstrings on modules, classes, and non-trivial functions.
- Prefer the standard library. Justify any optional dependency.
- Match the existing structure: tools go in `personal_ai/tools/`, integrations
  in `personal_ai/integrations/`.

## Adding a tool

The easiest way to add an ability is a plugin (no core changes needed). Drop a
file in `~/.personal_ai/plugins/`:

```python
from personal_ai.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "What it does."
    parameters = {"arg": "What this argument is."}
    def run(self, args):
        return "result"

def get_tools(config, memory):
    return [MyTool()]
```

To add a tool to the core, create it under `personal_ai/tools/`, register it in
`personal_ai/tools/__init__.py`, and add a test.

## Adding an integration

Subclass `Integration` in a module under `personal_ai/integrations/`, declare
`required_env`, implement `actions()` and `call()`, and register it in
`personal_ai/integrations/__init__.py`. Report `needs setup` until the required
credentials are present. Do not use scraping; use official APIs only.

## Pull request process

1. Create a branch off `main` (for example `fix-memory-trim` or `add-x-tool`).
2. Make your change with tests.
3. Run the test suite and the compile check.
4. Push your branch and open a pull request against `main`.
5. In the description, summarize the change, note what you tested, and call out
   any limitations.

Keep pull requests focused. Small, single-purpose changes are reviewed faster.

## Branch protection (for maintainers)

This repository is set up so changes land through pull requests. To enforce
that on GitHub:

1. Go to **Settings -> Branches** in the repository.
2. Under **Branch protection rules**, click **Add branch ruleset** (or
   **Add rule** in the classic UI).
3. Set the branch name pattern to `main`.
4. Enable these protections:
   - **Require a pull request before merging** (so nothing is pushed straight
     to `main`).
   - **Require status checks to pass before merging**, and select the **CI**
     check from the GitHub Actions workflow.
   - **Require branches to be up to date before merging.**
   - Optionally **Require approvals** (1 is a good default) and
     **Require conversation resolution before merging.**
5. Save the rule.

After this is enabled, push feature branches and merge through pull requests.
The CI workflow in `.github/workflows/ci.yml` runs the test suite on Python
3.9 through 3.12 for every push and pull request.
