# Provenance and attribution

Curated on 2026-10-25 from the locally installed AI Research Workbench v2 (installation dated 2026-10-15). The original package was AI-assisted, followed by local reporting fixes and workflow changes. This export retains experiment/import code and synthetic tests, clears notebook outputs before fresh default-off execution, and removes machine-specific acceptance text. The portable hub reuses the existing hub's styles, filtering, versioned-source links and note/export widgets with a new browser-storage key and public-only content.

The code inventory and file hashes are in `EXPORT_MANIFEST.json`. There is no blanket license for this code; public availability does not grant an MIT or other open-source license. Generated notebook-preview markup uses nbconvert/Pygments; their full BSD notices are retained in THIRD_PARTY_NOTICES.md. No other third-party implementation source, model weights, paper full text, PDF, artwork or actual research dataset is bundled. Dependencies are imported rather than vendored and retain their own licenses.

Method/reference attribution:

- DFT: *On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification*, [versioned paper](https://arxiv.org/html/2508.05629v3). The loss/reference examples illustrate the described weighting; they do not reproduce the paper's empirical results.
- On-policy distillation: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*, [versioned paper](https://arxiv.org/html/2306.13649v3). Linked as reading material; its implementation is not copied here.
- Optional training target: [Qwen3-0.6B-Base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base). No model/tokenizer files are included.
- Optional training APIs: [Transformers](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/model) and [PEFT LoRA](https://huggingface.co/docs/peft/v0.15.0/en/package_reference/lora). Referenced packages are separately installed and licensed.

The 12 training and 4 validation arithmetic rows are synthetic plumbing examples, not the Dolly teaching sample or personal experiment data. Bibliographic titles/authors/URLs identify external works and confer no rights to those works.
