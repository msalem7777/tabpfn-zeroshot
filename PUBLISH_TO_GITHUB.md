# Publish this folder to your GitHub

These instructions use Windows PowerShell. Each command is one line. They create a new remote repository; they do not deploy the app as a public web service.

## 1. Extract and open

Extract the supplied ZIP into a new folder. Open the inner `tabpfn-zeroshot` folder containing `README.md` and `pyproject.toml`. In File Explorer's address bar, type `powershell` and press Enter. Use this clean publication copy rather than copying your active app's data, papers, saved runs, API tokens or virtual environment into it.

## 2. Check Git and GitHub CLI

```powershell
git --version
gh --version
```

If missing, install [Git for Windows](https://git-scm.com/download/win) and [GitHub CLI](https://cli.github.com/), then reopen PowerShell in the project folder.

## 3. Sign in

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Complete the browser flow and check that the account shown is the one that should own the repository. Do not put a GitHub token in source files.

## 4. Make the initial local commit

```powershell
git init -b main
git add .
git status --short
git diff --cached --stat
```

Review the staged file list. This copy includes code, tests, documentation and synthetic fixtures; `.gitignore` excludes common data/export/credential locations. If you added files yourself, check them before committing. An ignore file does not remove files already tracked by Git.

```powershell
git commit -m "Initial literature-to-context TabPFN prototype"
```

If Git requests an author identity, set it for this repository using your actual name and preferred GitHub commit email (a GitHub-provided noreply address is fine), then repeat the commit:

```powershell
git config user.name "YOUR NAME"
git config user.email "YOUR COMMIT EMAIL"
```

## 5. Create the remote and push

Choose **one** command. Replace `tabpfn-zeroshot` if you prefer a different unused name.

Private:

```powershell
gh repo create tabpfn-zeroshot --private --source=. --remote=origin --push
```

Public, if you want anyone to see the code:

```powershell
gh repo create tabpfn-zeroshot --public --source=. --remote=origin --push
```

The command creates the GitHub repository, connects it as `origin`, and pushes your commit. Do not create another empty remote with the same name first. If the name already exists, select a new name; do not force-push over an existing project.

## 6. Open and verify

```powershell
gh repo view --web
git status
git remote -v
```

GitHub should display the README and render the equations. `git status` should say the working tree is clean. Check your chosen visibility on GitHub.

No project license has been chosen. You can publish privately first and decide what reuse license to grant before presenting this as an open-source release. Third-party package and model terms remain separate. This publication copy omits the borrowed mascot and uses a plain ZS mark.

## Later changes

From this same folder, after editing and reviewing:

```powershell
git add .
git diff --cached --stat
git commit -m "Describe the change"
git push
```

References: [GitHub local-code guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github), [gh repo create](https://cli.github.com/manual/gh_repo_create).
