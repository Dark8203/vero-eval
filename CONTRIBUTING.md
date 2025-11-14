# Contributing to Vero Eval

🎉 Thanks for your interest in contributing to **Vero-Eval** — an open-source framework for evaluating AI pipelines and LLM systems.  
We welcome all kinds of contributions — from bug fixes and documentation improvements to new metrics, reports, and test generators.

---

## 🧩 How to Contribute

We follow the **Fork → Branch → Pull Request (PR)** workflow (visual guide [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).

1. **Fork** this repository and create your branch from `main`.
2. **Clone** your fork locally.
3. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b new-feature # give any name you want;)
4. Create a **virtual environment using Python 3.12** (We recommend using uv to create venv and installing dependencies):
   ```bash
   uv venv .venv --python 3.12   #if using python : python -m venv venv
   source venv\Scripts\activate  # On Windows
   source venv/bin/activate      # On macOS/Linux
   ```
5. **Install** dependencies:
   ```bash
   uv pip install -r requirements.txt
   
6. Follow existing **code patterns** and include a clear **docstring** and references.

7. **Commit and push** your changes to your fork.

8. Open a **Pull Request** against the main branch with a clear title and description.

---

## 🪶 Documentation Improvements

Docs live on the **docs branch** under **/docs** directory. We use Markdown for static pages.

You can contribute by:

- Fixing typos or improving explanations.
- Adding usage examples.
- Writing how-to guides or tutorials.

> To update the docs, create a branch from `docs`, make your changes, and open a PR against the `docs` branch.

---

## 🧹 PR Guidelines

#### Before opening a PR, please:

- Check that your changes pass all tests.
- Ensure the code is formatted and linted correctly.
- Tag relevant maintainers or issues if applicable.

#### Guidelines for PRs:
- Keep PRs focused and small.
- Add or update tests for all new features.
- Update docs where applicable.

---

## 💬 Community & Support

- Ask questions or suggest ideas in GitHub Discussions

-  Report bugs via Issues

🤝 Join our [Discord](https://discord.gg/cbSAwJ3CUm) for real-time collaboration.

