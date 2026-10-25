# AI Research Workbench

Follow a question from a paper and its source code to an executable notebook example, then retain the configuration, logs and outputs of a run. Import a downloaded run archive into a local report, including records of failed runs.

Open `reading_hub.html` for selected paper links, versioned arXiv/DOI links, notebook previews and browser-local notes that export as Markdown. Notes are not uploaded. Serve the extracted repository locally if your browser restricts file downloads or browser storage; storage belongs to that browser origin.

## Start with the CPU examples

Use Python 3.12 with NumPy for the CPU contracts. Notebook execution also needs Jupyter, nbformat and nbconvert. These are requirements, not bundled environments. PyTorch is optional for the two autograd checks; absent PyTorch is reported as skipped. No dependency is installed by the hub or default notebook path.

```sh
python -m unittest discover -s tests -v
cd PostTraining
python -m unittest discover -s tests -v
python -m experiments.run_pair
```

The final command only reports preflight information by default. Open `PostTraining/notebooks/00_cpu_checks.ipynb` to inspect prompt masking, token loss and SFT/DFT weighting with small synthetic examples. Saved HTML previews are under `PostTraining/notebooks/previews/`.

## Carry a run back to a report

`PostTraining/notebooks/01_colab_sft_dft.ipynb` connects the same checks to a bounded SFT/DFT LoRA smoke comparison. `RUN_INSTALL` and `RUN_TRAIN` are **false** by default. Its download bundle is `PostTraining_Colab_v2.zip`, containing only this package's experiment source, tests, notebooks and synthetic data.

Actual training requires a separately prepared CUDA environment and explicit enabling of the training flag. `requirements-colab.txt` is a candidate dependency set, not a GPU-validated lockfile. Direct CLI training requires `--confirm-smoke-training`; the notebook's parent launcher additionally applies a 20-minute subprocess timeout. This package does not provide automatic mid-run resumption.

A run archive keeps source/data snapshots, model revision, configuration, console logs, status, and any outputs already produced. Only a complete pair yields a comparison. After downloading an archive, return to the extracted repository root:

```sh
python tools/research_files.py import-run /path/to/run.zip
```

Read `runs/<run-id>/RESULT.md` alongside the retained original evidence. Invalid archives and existing destinations are refused. Interrupted runtimes may not finish producing an archive: confirm the downloaded file rather than assuming a remote ZIP was saved locally. Archives can include input data and outputs; review their sharing rights before publishing them.

`PostTraining/templates/experiment_record.json` is a blank record. `python tools/research_files.py backup --output /chosen/path/backup.zip` makes a private lightweight backup; it is **not a public export tool** and may include your notes. It excludes model weights, environments and caches rather than sanitizing private text.

## Sources and scope

An AI-assisted export of a local workbench, with synthetic arithmetic examples, notebook checks and run-import utilities. It includes no trained weights or measured model improvements; CUDA training has not been validated. The selected paper links are dated references.

See [PROVENANCE.md](PROVENANCE.md) for source and attribution, and `EXPORT_MANIFEST.json` for the exact exported file inventory. No blanket source-code license is granted by this repository. Third-party packages and referenced papers retain their own licenses and rights.
