# Colab Setup Guide
### SLM Distillation Project — Break Through Tech AI Studio 2026

This guide walks you through everything from scratch. If you have never used Google Colab before, start at Step 1. If you already have a Google account and have used Colab, jump to Step 3.

**Time required:** ~30–45 minutes for first-time setup, then 5–8 minutes at the start of each session.

---

## What is Google Colab and Why We Use It

Google Colab is a free cloud service that gives you access to a computer with a GPU (graphics card) in your browser. You write Python code in a notebook interface — similar to Jupyter — and run it on Google's hardware, not your own laptop.

We use Colab because:
- Fine-tuning a language model requires a GPU with ~16 GB of memory
- Most laptops (including M1/M2 Macs) can run small models but are too slow for production-quality training
- Colab's T4 GPU trains our models 10–20× faster than a MacBook CPU

**Your code lives on GitHub. Your data and model weights live on Google Drive. Colab is just the computer you rent temporarily to run things.**

When a Colab session ends (after ~12 hours, or when you close the browser), the computer is wiped. But because code is on GitHub and outputs are on Drive, you never lose your work.

---

## What You Will Set Up

| What | Where | Purpose |
|------|-------|---------|
| Google account | gmail.com or your Cornell email | Access to Colab and Drive |
| Google Drive folder | drive.google.com | Store data, models, outputs |
| Anthropic API key | console.anthropic.com | Claude Haiku for label generation |
| HuggingFace token | huggingface.co | Download open-source models |
| GitHub PAT | github.com | Pull/push code between Colab and GitHub |
| Colab Secrets | colab.research.google.com | Store keys securely |
| Your project branch | GitHub | Your personal code workspace |

---

## Step 1 — Verify your Google Account

**Use your Cornell email (`netid@cornell.edu`) for everything in this project.** Cornell provides Google Workspace for Education, which gives you more Drive storage than a personal Gmail account.

1. Go to [gmail.com](https://gmail.com) and sign in with your Cornell email
2. If prompted, use your NetID and Cornell password
3. Once signed in, go to [drive.google.com](https://drive.google.com) to confirm it works

> **If you can only access Colab with a personal Gmail (not Cornell email):** That is also fine. Personal Gmail accounts have 15 GB of Drive storage, which is enough for Phase 1. Just be consistent — use the same Google account for Drive and Colab.

---

## Step 2 — Set Up Your Google Drive Folder Structure

1. Go to [drive.google.com](https://drive.google.com)
2. Click **+ New** → **New folder** → name it `slm-distillation`
3. Open the `slm-distillation` folder
4. Create these subfolders inside it (click **+ New** → **New folder** for each):
   - `data` — then inside `data`, create `raw`, `processed`, `checkpoints`
   - `outputs`
   - `hf_cache`

Your Drive should look like this:
```
My Drive/
└── slm-distillation/
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── checkpoints/
    ├── outputs/
    └── hf_cache/
```

> **Note:** The session setup notebook (Cell 3) also creates these folders automatically. But it's good to have them ready.

---

## Step 3 — Get Your Anthropic (Claude) API Key

Claude Haiku is used to generate the training labels (teacher LLM). You need your own key because API costs are billed per use.

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Click **Sign up** (top right) — use your Cornell email
3. Verify your email address
4. Once logged in, click **API Keys** in the left sidebar
5. Click **+ Create Key**
6. Name it: `slm-distillation-colab`
7. Click **Create Key**
8. **COPY THE KEY IMMEDIATELY** — it is shown only once and starts with `sk-ant-`
9. Store it somewhere safe (Notes app, password manager)

> **Cost for Phase 1:** Labeling ~20 clusters × 5 prompts × 2 (tokens in + out) costs approximately **$0.02–0.05 total**. Very affordable on the free tier.

---

## Step 4 — Get Your HuggingFace Token

HuggingFace hosts the open-source models we fine-tune. You need a token to download them.

1. Go to [huggingface.co](https://huggingface.co)
2. Click **Sign Up** — use your Cornell email
3. Verify your email
4. Once logged in, click your profile picture (top right) → **Settings**
5. In the left sidebar, click **Access Tokens**
6. Click **+ New token**
7. Name: `colab-slm`, Type: **Read** (not Write — Read is enough)
8. Click **Create token**
9. **COPY THE TOKEN** — starts with `hf_`
10. Store it safely

---

## Step 5 — Create a GitHub Personal Access Token (PAT)

The PAT lets Colab clone and push to your GitHub repository without a password.

1. Go to [github.com](https://github.com) and sign in
2. Click your **profile picture** (top right) → **Settings**
3. Scroll all the way down in the left sidebar → click **Developer settings**
4. Click **Personal access tokens** → **Tokens (classic)**
5. Click **Generate new token** → **Generate new token (classic)**
6. Fill in:
   - **Note:** `colab-slm-distillation`
   - **Expiration:** 90 days (this covers the whole program)
   - **Scopes:** check the box next to **repo** (this gives full repository access)
7. Click **Generate token** at the bottom
8. **COPY THE TOKEN** — starts with `ghp_` and shown only once
9. Store it safely

---

## Step 6 — Set Up Colab Secrets

Colab Secrets store your API keys securely. They are attached to your Google account and automatically available in any Colab notebook you open, without ever being saved in a file or committed to GitHub.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Open any notebook (or create a new one)
3. Look at the left sidebar — you will see several icons. Click the **key icon** (🔑) — it says "Secrets" when you hover over it
4. For each key below, click **+ Add new secret**:

| Secret name (type EXACTLY as shown) | Value |
|--------------------------------------|-------|
| `ANTHROPIC_API_KEY` | Your Claude key from Step 3 (starts with `sk-ant-`) |
| `HF_TOKEN` | Your HuggingFace token from Step 4 (starts with `hf_`) |
| `GITHUB_PAT` | Your GitHub token from Step 5 (starts with `ghp_`) |
| `OPENAI_API_KEY` | Your OpenAI key (optional — skip if you don't have one) |

> **Important:** The secret names must match exactly, including capitalization. The code reads them by exact name.

> **Security:** Secrets are stored in your Google account. They are NOT shared with others, NOT visible in notebooks, and NOT committed to GitHub. Never paste your keys directly into notebook cells.

---

## Step 7 — Create Your Student Branch on GitHub

Each pair of students works on their own branch. This keeps your experiments separate from other pairs and from the main codebase.

**Do this once** (not every session):

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Open the file `notebooks/00_session_setup.ipynb` from GitHub:
   - Click **File** → **Open notebook**
   - Click the **GitHub** tab
   - Enter the repository URL and find `notebooks/00_session_setup.ipynb`
3. Run **Cells 1 through 7** in order (click the ▶ play button on each cell)
4. Scroll to the bottom of the notebook to find **"First session only — create your branch"**
5. Edit `BRANCH_NAME` to your pair's branch name:
   - Format: `pair-X/model-name` (e.g., `pair-1/smollm2-1.7b`)
   - Ask your advisor which pair number you are and which model you're using
6. Run that cell

Your branch is now created on GitHub. In all future sessions, Cell 7 will automatically pull the latest version of your branch.

---

## Every Session — Running the Setup Notebook

At the start of every Colab session (including the day after your first session), follow these steps:

### Open the setup notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File** → **Open notebook** → **GitHub** tab
3. Paste the repository URL, find `notebooks/00_session_setup.ipynb`, open it

Alternatively, if you already have the notebook saved to Drive:
1. Click **File** → **Open notebook** → **Google Drive** tab
2. Find and open `00_session_setup.ipynb`

### Run the cells in order

Click the ▶ play button on each cell, waiting for one to finish before running the next:

| Cell | What it does | Time |
|------|-------------|------|
| Cell 1 — Verify GPU | Checks you have a T4 GPU. If not, reconnect. | instant |
| Cell 2 — Mount Drive | Opens a permission popup in your browser. Click Allow. | 15 sec |
| Cell 3 — Create folders | Creates the Drive directory structure (safe to re-run). | instant |
| Cell 4 — HF cache | Points model downloads to Drive so they persist. | instant |
| Cell 5 — Load secrets | Reads your API keys from Colab Secrets. | instant |
| Cell 6 — Install packages | Installs Python dependencies. **Takes 3–4 minutes.** | 3–4 min |
| Cell 7 — Clone/pull repo | Downloads or updates the code from GitHub. | 15 sec |
| Cell 8 — Verify setup | Checks that everything is ready. All boxes should show ✅. | instant |

**Total: 5–8 minutes.** After Cell 8 shows all green checkmarks, you are ready.

> **If Cell 1 shows K80 instead of T4:** Go to **Runtime** → **Disconnect and delete runtime**, then reconnect. T4 is the 16 GB GPU you need. K80 has only 12 GB and will cause out-of-memory errors during training.

---

## Running the Pipeline

After setup is complete, scroll down in the notebook to the "Run the pipeline" section.

### Full training run (most common)

This runs everything from scratch: clustering → labeling → fine-tuning → evaluation.

```
Expected time on Colab T4:
  Clustering (embeddings + UMAP + HDBSCAN): ~3 minutes
  Label generation (20 clusters × 5 prompts via Claude): ~5 minutes
  Fine-tuning (SmolLM2-1.7B, 3 epochs, ~250 examples): ~20–40 minutes
  Evaluation (inference + metrics + judge): ~10 minutes
  Total: ~40–60 minutes
```

Run the "Full training + evaluation pipeline" cell (or in the terminal):

```bash
python main.py \
    --phase 1 \
    --config configs/phase1_config.yaml \
    --device_mode colab
```

### Skipping steps (resuming after interruption)

If your session ended mid-run, most steps checkpoint their work. When you restart:

1. Run the setup notebook (Cells 1–8)
2. Check which files already exist on Drive:
   ```python
   import os
   drive = '/content/drive/MyDrive/slm-distillation'
   for f in ['data/processed/bitext_labeled.csv',
             'data/processed/train.jsonl']:
       exists = os.path.exists(f'{drive}/{f}')
       print(f'{"✅" if exists else "❌"} {f}')
   ```
3. In `configs/phase1_config.yaml`, set any completed steps to `false`:
   ```yaml
   pipeline:
     run_clustering:       false   # if bitext_clustered.csv exists
     run_preprocessing:    false   # if bitext_grouped.csv exists
     run_label_generation: false   # if bitext_labeled.csv exists
     run_finetuning:       true    # if lora_adapter/ doesn't exist yet
     run_baseline_eval:    true
     run_finetuned_eval:   true
     run_llm_judge:        true
     run_business_eval:    true
   ```
4. Run the pipeline again

### Evaluation only (model already trained)

If you want to re-run evaluations on a model you already trained, set `existing_run_dir` in the config and set all `run_*` flags to false except the evaluation ones:

```yaml
evaluation:
  existing_run_dir: "outputs/20260820_0014_SmolLM2-1.7B-Instruct_ep6"
pipeline:
  run_clustering:       false
  run_preprocessing:    false
  run_label_generation: false
  run_finetuning:       false
  run_baseline_eval:    true
  run_finetuned_eval:   true
  run_llm_judge:        true
  run_business_eval:    true
```

### Live demo mode

The demo loads models once and then accepts ticket files interactively.

1. Update `ADAPTER_DIR` in the demo cell to your run's adapter path
2. Run the "Live demo mode" cell
3. The program will print "Ready" and wait for input
4. Type a ticket file path (e.g., `demo_tickets/it_password_reset.txt`) and press Enter
5. Select a prompt (or press Enter to keep the current one)
6. Results appear immediately
7. Type `exit` to stop

> **Note:** In Colab, interactive input works in notebook cells but the experience is slightly different from a terminal. Each time you enter text, click the text box that appears below the cell.

---

## Switching Between Models

The config controls which model is used. To switch:

1. Open `configs/phase1_config.yaml`
2. Change `student_slm.model_id`:
   ```yaml
   student_slm:
     model_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct"   # 1.7B — recommended
     # model_id: "microsoft/Phi-3.5-mini-instruct"        # 3.8B — primary Phase 1
     # model_id: "HuggingFaceTB/SmolLM2-360M-Instruct"   # 0.36B — fastest
   ```
3. Push the change to GitHub:
   ```bash
   git add configs/phase1_config.yaml
   git commit -m "Switch to SmolLM2-1.7B"
   git push
   ```
4. Pull the change in your next Colab session (Cell 7 handles this automatically)

> **The first time a new model is used, it downloads to Drive (`hf_cache/`). This takes 2–10 minutes depending on size. Every subsequent session loads from Drive in ~30 seconds.**

---

## Understanding Colab Limits

### Session length
Free Colab sessions last up to **12 hours**. After that, the session is automatically terminated. Your Drive files are safe — only the temporary Colab disk is wiped.

### Idle timeout
If no code is running and you are idle for ~90 minutes, Colab disconnects you. The session also ends if you close your browser tab while code is running. **Training runs to completion as long as the tab is open or code is executing** — you don't need to watch it.

### Weekly GPU quota
Free Colab does not publish an exact quota, but typical usage allows 15–30 hours of T4 GPU time per week. One full training run uses ~1 hour. You can do 3–5 experiments per week comfortably on the free tier.

### What to do if you run out of quota
- Wait until the next day (quota partially resets)
- Switch to CPU mode for non-training steps (data processing, evaluation)
- Disconnect and reconnect to try getting a different session

### Checking your remaining quota
Go to **Runtime** → **View resources** in Colab to see current session usage.

---

## File Organisation Summary

Understanding what lives where prevents confusion:

| What | Where | Backed up? |
|------|-------|-----------|
| Python code | GitHub | ✅ Yes — push often |
| Config files | GitHub | ✅ Yes |
| Demo ticket files | GitHub | ✅ Yes |
| Bitext raw data | Drive: `data/raw/` | ✅ Redownloadable |
| Clustered/labeled CSVs | Drive: `data/processed/` | ✅ Recomputable |
| Clustering checkpoints (pkl) | Drive: `data/checkpoints/` | ✅ Saves time |
| LoRA adapter weights | Drive: `outputs/{run_id}/models/` | ✅ Keep these! |
| Evaluation CSVs | Drive: `outputs/{run_id}/evaluation/` | ✅ |
| HuggingFace models | Drive: `hf_cache/` | ✅ Saves re-downloading |
| Intermediate eval files | Drive: `outputs/{run_id}/evaluation/_cache/` | ✅ |

---

## Saving Your Work to GitHub

After a successful run, save your changes (config tweaks, notes, analysis notebooks):

```bash
git add .
git commit -m "Describe what you did: e.g. SmolLM2-1.7B ep3, cosine 0.76"
git push
```

Run this in the "Save your code changes to GitHub" cell at the bottom of the setup notebook.

**Important:** Do NOT push model weights or data files. The `.gitignore` handles this automatically — anything in `outputs/`, `data/`, and `hf_cache/` is excluded.

---

## Pulling Advisor Updates

Your advisor may push bug fixes or improvements to the `main` branch. Pull them into your branch:

```bash
git fetch origin
git merge origin/main
```

If there are conflicts (unlikely but possible), ask your advisor for help resolving them.

---

## Common Issues and Fixes

**"CUDA out of memory" during training**
- Reduce `per_device_train_batch_size` from 4 to 2 in the config
- Make sure `gradient_checkpointing: true`
- Disconnect and reconnect to get a fresh T4 (sometimes a T4 has other processes using memory)

**"No module named X" error**
- You forgot to run Cell 6 (install dependencies) in the setup notebook
- Re-run Cell 6

**GPU shows K80 instead of T4**
- Go to **Runtime** → **Disconnect and delete runtime**
- Reconnect — you'll get a different machine assignment
- Check Cell 1 again

**Session ended while training was running**
- Your progress is NOT lost — checkpoints were saved to Drive
- Start a new session, run setup cells, check which steps completed, set completed steps to `false` in config, re-run

**"GITHUB_PAT not found" error**
- Check the 🔑 Secrets panel — the name must be exactly `GITHUB_PAT`
- Secrets are per-Google-account — if you switched accounts, re-add them

**"ANTHROPIC_API_KEY not set" error**
- Check the 🔑 Secrets panel — the name must be exactly `ANTHROPIC_API_KEY`
- Make sure the toggle next to the secret is switched ON for the current notebook

**Files on Drive not visible from Colab**
- Re-run Cell 2 (mount Drive) — sometimes Drive disconnects
- If that doesn't work: **Runtime** → **Restart runtime** and re-run all setup cells

**HuggingFace model download is slow (taking 10+ minutes)**
- This happens on first download — subsequent sessions load from Drive cache
- Do not interrupt the download — let it complete

**`bitext_labeled.csv` exists but teacher latency shows 0 in business eval**
- This is expected when labels were loaded from the checkpoint (not re-generated)
- To re-generate labels with fresh timing: delete `data/processed/bitext_labeled.csv` from Drive and re-run with `run_label_generation: true`

---

## Quick Reference Card

Print this or keep it open during sessions.

```
EVERY SESSION:
1. Open colab.research.google.com
2. Open notebooks/00_session_setup.ipynb
3. Run Cells 1 → 8 (wait for each to finish)
4. All ✅ in Cell 8? → Ready to run

FULL PIPELINE:
python main.py --phase 1 --config configs/phase1_config.yaml --device_mode colab

DEMO MODE:
python main.py --phase 1 --config configs/phase1_config.yaml --device_mode colab \
    --mode demo --adapter_dir outputs/YOUR_RUN/models/lora_adapter

SAVE TO GITHUB:
git add . && git commit -m "message" && git push

GET ADVISOR UPDATES:
git fetch origin && git merge origin/main
```

---

## Getting Help

- **Code issues / bugs:** Post in the project Slack channel with the full error message
- **Colab / Drive issues:** Ask your coach
- **API key issues:** Re-check the setup steps above carefully — most key errors are typos in the secret name
- **Model quality questions / research questions:** Ask your advisor in office hours
