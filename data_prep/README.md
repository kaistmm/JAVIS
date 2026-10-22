# Training data

All datasets go through one pipeline; only the first step (turning a dataset into manifests) is dataset specific.

```
videos/wavs ─► datasets/<set>.py ─► clips.jsonl (+ dialogues.jsonl)
                                     │
            extract_mcoeff.py ───────┤  FLOAT 20-d motion per clip (25 fps)  ─► pack_mcoeff.py ─► store/{feats.bin,index.jsonl,meta.json}
            extract_speech_tokens.py ┘  CosyVoice3 speech tokens (25 Hz)
                                     │
            build_stage1_rows.py / build_stage2_rows.py ─► rows/*.jsonl (LLaMA-Factory sharegpt rows)
            compute_mcoeff_stats.py (Talker-T2AV store) · build_gcond_tables.py (every store)
```

## Recipes

Common steps for one set of clips (`<set>`); `--no_crop` only for videos that are already 512x512 face crops, and
`--video_list` (one video path per line) only to restrict the extraction to some of the videos under `--dir_path`:
```bash
python data_prep/extract_mcoeff.py --dir_path <video root> [--video_list videos.txt] --output_dir data/<set>/mc [--no_crop] --stats_jsonl
python data_prep/pack_mcoeff.py --clips data/<set>/clips.jsonl --video_root <video root> --mcoeff_dir data/<set>/mc/mcoeffs --out data/<set>/store
python data_prep/extract_speech_tokens.py --manifest data/<set>/clips.jsonl --output_dir data/<set>/tokens --speech_cpu_threads 8
```

**Stage 1, Talker-T2AV** (motion is extracted for every clip first: the quality filter reads its statistics)
```bash
python data_prep/extract_mcoeff.py --dir_path <Talker-T2AV-Data>/video --output_dir data/talker/mc --no_crop --stats_jsonl
python data_prep/datasets/talker_t2av.py filter   --root <Talker-T2AV-Data> --mcoeff_out data/talker/mc --out data/talker/exclude.txt
python data_prep/datasets/talker_t2av.py manifest --root <Talker-T2AV-Data> --exclude data/talker/exclude.txt --out data/talker/clips.jsonl
python data_prep/pack_mcoeff.py --clips data/talker/clips.jsonl --video_root <Talker-T2AV-Data>/video --mcoeff_dir data/talker/mc/mcoeffs --out data/talker/store
python data_prep/extract_speech_tokens.py --manifest data/talker/clips.jsonl --output_dir data/talker/tokens --speech_cpu_threads 8
python data_prep/compute_mcoeff_stats.py --stores data/talker/store --out data/talker/stats.pt
python data_prep/build_gcond_tables.py --stores data/talker/store
python data_prep/build_stage1_rows.py --clips data/talker/clips.jsonl --tokens data/talker/tokens/speech_tokens.jsonl \
    --store data/talker/store --out_dir data/talker/rows
```

**Stage 2, Seamless Interaction** (`train` for training, `dev` for validation)
```bash
for S in train dev; do
  python data_prep/datasets/seamless_download.py --root <dl root> --label improvised --split $S
  for P in manifest segment slice; do
    python data_prep/datasets/seamless_segment.py $P --split $S --src <dl root> --assets <seamless_interaction>/assets --out_root data/seamless/$S
  done
  python data_prep/datasets/seamless_manifest.py --out_root data/seamless/$S --split $S --out_dir data/seamless/$S
  # (common steps WITH face cropping, into data/seamless/$S/{mc,store,tokens})
done
python data_prep/build_stage2_rows.py --dialogues data/seamless/train/dialogues.jsonl --clips data/seamless/train/clips.jsonl \
    --tokens data/seamless/train/tokens/speech_tokens.jsonl --store data/seamless/train/store --out data/seamless/rows/train.jsonl \
    --episode_break --min_target_frames 50
python data_prep/build_stage2_rows.py --dialogues data/seamless/dev/dialogues.jsonl --clips data/seamless/dev/clips.jsonl \
    --tokens data/seamless/dev/tokens/speech_tokens.jsonl --store data/seamless/dev/store --out data/seamless/rows/val.jsonl \
    --episode_break --min_target_frames 50
```

**Stage 2, MultiDialog** (`train` for training, `valid_freq` for validation)
```bash
python data_prep/datasets/multidialog.py --root <MultiDialog> --video_dir <MultiDialog>/video_25fps --out_dir data/multidialog
for S in train valid_freq; do
  # (common steps WITH face cropping, from data/multidialog/clips_$S.jsonl into data/multidialog/$S/{mc,store,tokens})
  python data_prep/build_stage2_rows.py --dialogues data/multidialog/dialogues_$S.jsonl --clips data/multidialog/clips_$S.jsonl \
      --tokens data/multidialog/$S/tokens/speech_tokens.jsonl --store data/multidialog/$S/store --out data/multidialog/rows/$S.jsonl
done
mv data/multidialog/rows/valid_freq.jsonl data/multidialog/rows/val.jsonl
python data_prep/build_gcond_tables.py --stores data/seamless/{train,dev}/store data/multidialog/{train,valid_freq}/store
```
The rows land where `training/data/dataset_info.json` expects them (`data/<set>/rows/{train,val}.jsonl`).

## Manifests

`clips.jsonl`, one utterance clip per line:

| key | |
|---|---|
| `id` | unique clip id |
| `video` | 25 fps video of the speaker (the mcoeff source) |
| `wav` | the clip's speech |
| `text` | transcript (the assistant target text) |
| `split` | stage 1: `train` / `val` |
| `group`, `turn` | same person in the same video, utterance order (for the sibling gcond tables) |

`dialogues.jsonl`, one conversation per line: `{"id", "turns": [{"clip", "speaker", "episode"}], "assistant": [speakers]}`.
Each listed assistant speaker produces one pass over the dialogue; context user turns are audio, context assistant turns
are text, and the target is an assistant turn with speech tokens and motion (up to 3 context turns).

## Details that matter

- **Face crop.** Videos that are not already a tight 512x512 face crop must be extracted with face cropping (the default);
  Talker-T2AV clips are already cropped (`--no_crop`). Inference crops the portrait with the same rule.
- **Alignment.** A target uses M = min(#speech tokens, #motion frames) frames; clips where the two differ by more than 5
  frames are dropped as mispaired.
- **Normalization.** `compute_mcoeff_stats.py` on the Talker-T2AV store gives the decoder's mean/std; it is stored in
  the checkpoint and reused for stage 2.
- **gcond.** The decoder's global condition is the middle frame of another clip of the same person in the same video, at
  least two utterances away (`build_gcond_tables.py` writes `<store>/gcond.pt`); a clip with no such sibling uses its own
  middle frame.
- **Speech tokens.** The ONNX CUDA provider is pathologically slow on some machines; `--speech_cpu_threads 8` with
  many processes is usually faster.

## Datasets used for JAVIS

| set | script | notes |
|---|---|---|
| Talker-T2AV | `datasets/talker_t2av.py` | stage 1; `filter` removes 2.45% of clips (A/V length mismatch, scene cuts, static faces, no speech) |
| Seamless Interaction (improvised) | `datasets/seamless_download.py`, `seamless_segment.py`, `seamless_manifest.py` | full-duplex → half-duplex turns; build rows with `--episode_break --min_target_frames 50` |
| MultiDialog | `datasets/multidialog.py` | 25 fps videos; train with both speakers as assistant, validate on `valid_freq` |
| JAVIS-Syn | — | 334 h synthetic emotional dialogues (not released yet) |
