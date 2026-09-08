# Phase 0 Validation Results

Chu Han (Chinese-derived) character recognition, CHAT (pretrained, as-is) vs. NomNaOCR (pretrained CRNNxCTC, as-is), on the same sample pages.

**Classification of Chu Han vs. Chu Nom-proper is a Unihan-kMandarin heuristic, not verified ground truth - see build_hanzi_charset.py.**

## Aggregate

| Model | Correct Han chars | Total Han chars | Accuracy |
|---|---|---|---|
| CHAT (pretrained) | 151 | 679 | 22.2% |
| NomNaOCR (CRNNxCTC, pretrained) | 589 | 679 | 86.7% |
| CHAT (fine-tuned trial) | 28 | 679 | 4.1% |

## Per page

| Page | Total Han | CHAT correct | CHAT fine-tuned correct | NomNaOCR correct |
|---|---|---|---|---|
| DVSKTT1Quyenthu_DVSKTT_thu_III_1b | 65 | 29 | 1 | 51 |
| DVSKTT2Ngoaikytoanthu_DVSKTT_ngoai_I_3a | 71 | 2 | 1 | 48 |
| DVSKTT3Bankytoanthu_DVSKTT_ban_toan_II_22a | 73 | 13 | 5 | 67 |
| DVSKTT3Bankytoanthu_DVSKTT_ban_toan_I_23a | 33 | 6 | 1 | 29 |
| DVSKTT3Bankytoanthu_DVSKTT_ban_toan_VII_45b | 52 | 1 | 2 | 47 |
| DVSKTT3Bankytoanthu_DVSKTT_ban_toan_V_51b | 37 | 16 | 4 | 32 |
| DVSKTT4Bankythucluc_DVSKTT_ban_thuc_XIII_41b | 52 | 13 | 1 | 49 |
| DVSKTT4Bankythucluc_DVSKTT_ban_thuc_XIV_27a | 54 | 27 | 9 | 51 |
| DVSKTT4Bankythucluc_DVSKTT_ban_thuc_XV_2a | 34 | 0 | 1 | 31 |
| DVSKTT5Bankytucbien_DVSKTT_ban_tuc_XVIII_24b | 54 | 6 | 3 | 51 |
| DVSKTT5Bankytucbien_DVSKTT_ban_tuc_XVII_9b | 52 | 13 | 0 | 45 |
| LucVanTien_nlvnpf-0059-086 | 20 | 11 | 0 | 18 |
| TaleofKieu1866_page059a | 29 | 4 | 0 | 27 |
| TaleofKieu1871_page084 | 22 | 7 | 0 | 20 |
| TaleofKieu1872_page26b | 31 | 3 | 0 | 23 |

## Reading this result

- Scoring is a bag-of-characters comparison per page (see compare_hanzi.py docstring), not full Sequence/Character Accuracy or CER - those come in Phase 2 for a properly evaluated fine-tuned model.
- "CHAT (fine-tuned trial)" is a quick directional check (10 epochs on ~2000 training lines from NomNaOCR's own training split), not Phase 1's full fine-tuning pipeline - see README.md's "Fine-tuning trial" section for the full setup and conclusion.