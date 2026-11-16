# Driver distraction detection

Final project of COMP6826001 Deep Learning, BINUS University, academic year 2026/2027.
The system detects driver distraction with an RGB camera, live.
The deliverables are a working application and a technical report.

## Read first

- `docs/PLAN.md`: the plan, the gates, and the decisions of the owner.
- `docs/LITERATURE.md`: the papers with their methods, splits, metrics, and scores.
- `docs/starter/guideline.docx`: the course guideline. It is the authority on the deliverables.
- `docs/starter/BRIEF.md`: the way of work and the lessons from the earlier project.
- `docs/AI_LOG.md`: the AI usage log.

The folder `docs/starter/` and the file `docs/AI_LOG.md` are not in git.

## State on 2026-11-15

- Phase 0 is complete. R1 passed gate G1. See `results/replication/scores.md`.
- Phase 2 started after a yes of the owner. The runs use the Colab L4, one run at a time (decision D13).
- The public ViT-DD weights, face crops, and emotion labels do not download. See `results/vitdd_public_files.md`.
- The owner selected StateFarm as the only dataset. AUC v2 is cancelled (decision D7).
- The replication target is R1, the standard ViT of the ViT-DD paper, with the public split by driver (decision D9).
- The development folds come from `splits/statefarm_v2.json` (decision D8).

## Way of work

1. Replicate the paper result first. `docs/PLAN.md` gives the target and the gate.
2. If the replication passes the gate, start the development phase.
3. If the replication fails the gate, stop. Report the numbers to the owner. The owner decides the next step.
4. Do the smallest change that fully does the task. Use the code of the repository before new code.

## Fixed rules

- Never write, estimate, or fill in a metric, a training time, or a result. Every number comes from a saved result file. Write `TBD` until a result exists.
- The owner decides the class mapping, the labels, the targets, and each change to the experiment design.
- Split by driver before all other data work. Keep the tests for this rule green.
- The test drivers are locked. `docs/PLAN.md` names the only two points that score them.
- Fit thresholds and other statistics on training or validation drivers only.
- Do not put dataset images, or data derived from them, in git or in public material.
- Do not record a person in a car that moves. Record in a parked car only.
- Add an entry to `docs/AI_LOG.md` for each session that makes project material.

## Work with the owner

- Reply to the owner in Indonesian. Write documents and code comments in plain English.
- Decide the choices that the facts can compare. Record each such decision in `docs/PLAN.md`.
- Ask the owner about each large choice with a multiple-choice question.
- Each question gives a recommendation, and the gain and the loss of each option.
- Tell the owner what was not done and what was not checked.

## Data rules

- StateFarm comes from Kaggle under the competition rules. Do not share its files.
- Own recordings need the consent of each person. They stay private.
- Map the labels of own recordings to the StateFarm codes by class name.
- Raw data lives in `data/`. Run output lives in `runs/`. These two folders are not in git.

## Code rules

- Core logic lives in the Python package `ddd` in git.
- A notebook only mounts Drive, pulls the repository, installs the package, and starts scripts.
- Each run reads one config file.
- Each run folder holds the config, the metrics, the history of each epoch, the GPU type, the library versions, and the git commit.
- Save a checkpoint after each epoch. A run must continue after a disconnect.
- Start long jobs in the background and write the logs to Drive.
- New logic with a branch, a loop, or a metric gets one small test.

## Documents

- A hook lints each Markdown file for plain English.
- Keep each sentence at 25 words or fewer. End each table cell and each list item with a period.
- Use the word "check" only, and the word "config" only. The linter counts their synonyms as errors.

## Commands

Run the commands from the root of the repository. On Windows, use `.venv/Scripts/python` for `python`.

```
python -m venv .venv
python -m pip install -e ".[test,train]"
python -m pytest -q
python scripts/make_split.py
python scripts/audit_statefarm.py
python scripts/make_manifest.py data/raw/statefarm data/manifests/statefarm.csv
python scripts/make_manifest.py data/raw/statefarm data/manifests/statefarm.csv --check
python scripts/train.py configs/r1_vit.json
python scripts/train.py configs/b0.json --fold 0
python scripts/run_queue.py configs/queue_phase2.txt --runs runs --logs logs
python scripts/summarize_folds.py "G:/My Drive/driver-distraction/runs" b0
python scripts/score_gray.py "G:/My Drive/driver-distraction/runs" b0
python scripts/measure_speed.py configs/e1_deit3_small.json configs/e1_resnet50.json
python scripts/record_speed.py runs/RUN_NAME --key laptop
```

- `make_split.py` reads the public ViT-DD lists in `data/external/vitdd/annotations/SFDDD/`.
- `make_split.py` writes split v2. Use `--version v1` for split v1.
- `audit_statefarm.py` expects `data/raw/statefarm/driver_imgs_list.csv` and the class folders in `data/raw/statefarm/imgs/train/`.
- Use `--images` for another folder of class folders.
- The manifest lists image names. It stays in `data/` and not in git.
- `notebooks/colab_runner.ipynb` starts the scripts on Colab.
- `train.py` continues a run from `last.pt` in its run folder.
- The option `--fold` of `train.py` replaces the fold of the config. The run folder holds the config with the fold.
- `run_queue.py` trains the runs of a queue file one at a time. It skips each run with a `metrics.json`.
- `summarize_folds.py` copies the files of the 3 fold runs to `results/development/` and writes their mean scores.
- `score_gray.py` scores the 3 fold runs on their validation drivers in color and in gray scale. It writes the tables of each class and each driver too.
- `measure_speed.py` measures the processed frames per second of each config on this computer. It writes `results/development/speed_laptop.json`.
- The options `--epochs`, `--max-steps`, and `--batch-size` of `train.py` are for speed measurements only.
- Code that reads the test drivers calls `ddd.splits.drivers` with `unlock_test=True` and a reason.

## Colab

The AI runs the Colab work through the Colab MCP server. Do not ask the owner to run notebook cells.

1. Call `open_colab_browser_connection`. Then read the open notebook with `get_cells`.
2. Run `!nvidia-smi` first. If the runtime has no GPU, ask the owner to change the runtime type to a GPU.
3. Copy the cells of `notebooks/colab_runner.ipynb` into the open notebook, and run them in order.
4. Start each job of more than a few minutes in the background with `nohup`. Write its log to Drive.
5. Read the log later with a short cell.
6. While a job runs, run a short check cell at least every 30 minutes (decision D14). Colab can stop an idle runtime.
7. Copy each result file from Drive into `results/` in git. Copy the saved values only.

- The owner does only the steps that the AI cannot do. These steps are to open a notebook, pick the GPU, allow the Drive mount, and read the compute units.
- A fix in a Colab cell goes into `notebooks/colab_runner.ipynb` too. Keep the two the same.
- If the Colab tools are missing, check the server status first.
- If the server failed with a timeout, the owner runs its `uvx` command once in a terminal. Then the owner presses Reconnect in `/mcp`.
- After the work, tell the owner to stop the runtime. An open GPU runtime uses compute units.
- The Drive folder is `MyDrive/driver-distraction`, with `data/raw/statefarm/`, `manifests/`, `logs/`, `runs/`, and `results/`.
