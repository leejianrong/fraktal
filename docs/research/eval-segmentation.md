# Segmentation for eye-level evaluation (FRK-145)

Why: simple colour and edge statistics barely registered that our dense-urban strip is the wrong kind of
street for Manila-Quezon (FRK-144). Class shares (building, road, sky, vegetation, vehicle, person) from a
pretrained segmenter say it directly. The same model runs on the reference photos and on our frames, so
its own errors affect both sides alike.

Use: **evaluation and looking only.** The photos stay under `data/eval` and are never trained on or
shipped (ADR-0012). The class shares are internal numbers; no model output enters a dataset.

## What was read (2026-11-12)

Raw files fetched with `curl`, not summaries, except where marked.

| Item | Licence read | Verdict |
|---|---|---|
| OneFormer code (`SHI-Labs/OneFormer`) | LICENSE is the MIT text, copyright SHI Labs; README badge MIT | allowed |
| OneFormer weights `shi-labs/oneformer_coco_swin_large` | Hugging Face metadata `license: mit`, dataset COCO. No separate weights licence file in the repository (metadata only) | allowed for evaluation |
| COCO panoptic labels | Annotations are CC BY 4.0 (from memory, not re-read); images are Flickr photos under mixed licences. Evaluation use only | acceptable here |
| OneFormer ADE20K checkpoints | MIT metadata, but ADE20K's terms of use (read): "non-commercial research and educational purposes only" | **not used**: the dataset terms are the weaker link, and the COCO checkpoint covers the same street classes |
| SegFormer (`NVlabs/SegFormer`) | NVIDIA Source Code License, section 3.3: the Work may be used non-commercially, "for research or evaluation purposes only" | not used: same reason as nvdiffrast in ADR-0013 |
| Mask2Former code | LICENSE is MIT (Meta). The Hugging Face checkpoints (`facebook/mask2former-*`) say `license: other` and ship no licence text | not used until the weights terms are found |
| `transformers` and `scipy` libraries | Apache-2.0 and BSD (not re-read) | allowed as an optional extra, not a core dependency |

Loading check: `from_pretrained` reports 2 missing keys (the final Swin layer norm, unused for features)
and 24 unexpected position-index buffers. Both are harmless; masks look right by eye (people, vehicles,
road, pavement, trees, sky and facades on Nairobi and Manila-Quezon photos).

Cost: about 14 s per image on 16 CPU cores, 38 s to load, 7 GB RAM tight. A city sample (13 photos plus
40 frames) takes roughly 12 minutes; results are cached per image.

On a GPU it is about 11 times faster: 0.42 s per image against 4.8 s on 20 CPU cores (RTX 3080 Ti Laptop, 512 px
frames, float32, 1.55 GiB of GPU memory, 2026-11-14). The model picks `cuda` when torch sees a GPU;
`FRAKTAL_SEG_DEVICE=cpu` forces the CPU, and `cuda` without a GPU is an error. Class shares from the two devices
agreed to rounding on 15 frames, so one cache serves both.

## Limits

- COCO "stuff" classes are coarse: `building-other-merged`, `pavement-merged`, `tree-merged`. We group
  them (building, road, pavement, sky, vegetation, vehicle, person, fence, pole or sign).
- It sees roads and pavements by appearance, so a dusty dirt road may be labelled dirt or ground and a
  sealed road labelled road; the road group includes dirt, gravel and sand for that reason.
- Wires, signs and stalls are not classes. Those gaps are found by eye on the sheet.

## Dependencies added (optional group `seg`, `make install-seg`)

`uv lock` added `transformers`, `huggingface-hub`, `hf-xet`, `safetensors`, `tokenizers`, `scipy`,
`pydantic`, `httpx`, `typer`, `rich`, `pyyaml`, `regex`, `tqdm` and small helpers. Licences read from the
installed metadata: Apache-2.0, MIT, BSD, ISC; `tqdm` and `certifi` are MPL-2.0 (file-level copyleft,
unmodified, fine for an optional evaluation group). The default `make install` does not install the group.

## First result (2026-11-12): dense-urban strip against Manila-Quezon

12 photos, 20 frames, same model on both. Class shares, photos against synthetic: vegetation 34% / 0.5%,
building 5% / 49%, road 27% / 5%, pavement 2% / 34%, vehicle 9% / 0%, fence 5% / 0%.
Reading it: the dense-urban biome is the wrong kind of street for Manila-Quezon (a tall canyon, no trees),
and our flat grey road reads as pavement to the model, since it has no markings or kerbs. The z column
under-reports the vegetation gap (z -1.8) because the photos vary a lot; read the shares first.
