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

## Limits

- COCO "stuff" classes are coarse: `building-other-merged`, `pavement-merged`, `tree-merged`. We group
  them (building, road, pavement, sky, vegetation, vehicle, person, fence, pole or sign).
- It sees roads and pavements by appearance, so a dusty dirt road may be labelled dirt or ground and a
  sealed road labelled road; the road group includes dirt, gravel and sand for that reason.
- Wires, signs and stalls are not classes. Those gaps are found by eye on the sheet.
