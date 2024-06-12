# RNA Secondary Structure Datasets

## Rfam split

Want to have several splits to test generalization across families. For a given RNA family in ArchiveII, we want to avoid having related Rfam families in training. Tentatively, propose this split:

tRNA (https://rfam.org/search?q=tRNA%20AND%20entry_type:%22Family%22) and tmRNA (https://rfam.org/search?q=tmRNA)

- `CL00001`: `RF00005 (tRNA)`, `RF01852 (tRNA-Sec)`, `RF00023 (tmRNA)`, `RF02544 (mt-tmRNA)`, `RF01851 (cyano_tmRNA)`, `RF01850 (beta_tmRNA)`, `RF01849 (alpha_tmRNA)`
- `RF00233 (Tymo_tRNA-like)`
- `RF01075 (TLS-PK1)`
- `RF01077 (TLS-PK2)`
- `RF01084 (TLS-PK3)`
- `RF01085 (TLS-PK4)`
- `RF01088 (TLS-PK5)`
- `RF01101 (TLS-PK6)`

SRP RNA (https://rfam.org/search?q=signal%20recognition%20AND%20entry_type:%22Family%22)

- `CL00003`: `RF00169 (Bacteria_small_SRP)`, `RF01854 (Bacteria_large_SRP)`, `RF00017 (Metazoa_SRP)`, `RF01855 (Plant_SRP)`, `RF01857 (Archaea_SRP)`, `RF01502 (Fungi_SRP)`, `RF01856 (Protozoa_SRP)`, `RF01570 (Dictyostelium_SRP)`, `RF04183 (Bacteroidales_small_SRP)`

telomerase RNA (https://rfam.org/search?q=telomerase%20RNA%20AND%20entry_type:%22Family%22)

- `CL00004`: `RF01050 (Sacc_telomerase)`, `RF00024 (Telomerase-vert)`, `RF00025 (Telomerase-cil)`, `RF02462 (Telomerase_Asco)`

5S rRNA (https://rfam.org/search?q=5S%20rRNA%20AND%20entry_type:%22Family%22)

- `CL00113`: `RF00001 (5S_rRNA)`, `RF02547 (mtPerm-5S)`
- `RF02555 (hveRNA)`
- `RF02554 (ppoRNA)`

RNase P (https://rfam.org/search?q=RNase%20P%20RNA%20AND%20entry_type:%22Family%22)

- `CL00002`: `RF00010 (RNaseP_bact_a)`, `RF00009 (RNaseP_nuc)`, `RF00011 (RNaseP_bact_b)`, `RF00373 (RNaseP_arch)`, `RF02357 (RNaseP-T)`, `RF00030 (RNase_MRP)`, `RF01577 (RNase_P)`

Group 1 and 2 introns (https://rfam.org/search?q=Intron%20AND%20entry_type:%22Family%22)

- `CL00102`: `RF01998 (group-II-D1D4-1)`, `RF01999 (group-II-D1D4-2)`, `RF02001 (group-II-D1D4-3)`, `RF02003 (group-II-D1D4-4)`, `RF02004 (group-II-D1D4-5)`, `RF02005 (group-II-D1D4-6)`, `RF02012 (group-II-D1D4-7)`
- `RF00029 (Intron_gpII)`
- `RF00028 (Intron_gpI)`

23S rRNA:

- `CL00112`: `RF00002 (5_8S_rRNA)`, `RF02540 (LSU_rRNA_archaea)`, `RF02541 (LSU_rRNA_bacteria)`, `RF02543 (LSU_rRNA_eukarya)`, `RF02546 (LSU_trypano_mito)`

13S rRNA:

- `RF01959 (SSU_rRNA_archaea)`, `RF00177 (SSU_rRNA_bacteria)`, `RF01960 (SSU_rRNA_eukarya)`, `RF02542 (SSU_rRNA_microsporidia)`, `RF02545 (SSU_trypano_mito)`

## Pre-generated splits

The pre-generated splits (`test_split_1.lst, ..., test_split_9.lst`) are constructed using `python process_rna_sdb.py --random_state=0`
The output of the command:
```bash
python process_rna_sdb.py --random_state=0

Loading Rfamseq...
4170it [00:16, 253.68it/s]
Rfamseq loaded. Statistics:
Total of sequences: 3,117,783
Number of sequences with non-canonical bases: 13,382 (0.4%)
Number of sequences with canonical bases: 3104401 (99.6%)
Bases occuring in the dataset: {'S', 'R', 'D', 'K', 'B', 'H', 'G', 'Y', 'A', 'W', 'V', 'U', 'C', 'T', 'M', 'N'} (16 bases)
Number of families: 4,170
Total number of Rfam families reserved for testing: 1908
Total of sequences in reserved for testing: 15,830 (0.5%)

Split split_1
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,254
Number of test families: 1,916
Train ratio: 53.2%
Statistics after balancing:
Number of train families: 2,254
Number of test families: 1,916
Train ratio: 53.2%

Split split_2
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,254
Number of test families: 1,916
Train ratio: 97.9%
Statistics after balancing:
Number of train families: 2,020
Number of test families: 2,150
Train ratio: 88.5%

Split split_3
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,259
Number of test families: 1,911
Train ratio: 99.5%
Statistics after balancing:
Number of train families: 1,826
Number of test families: 2,344
Train ratio: 88.9%

Split split_4
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,261
Number of test families: 1,909
Train ratio: 95.0%
Statistics after balancing:
Number of train families: 1,977
Number of test families: 2,193
Train ratio: 87.4%

Split split_5
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,256
Number of test families: 1,914
Train ratio: 99.2%
Statistics after balancing:
Number of train families: 1,915
Number of test families: 2,255
Train ratio: 88.9%

Split split_6
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,253
Number of test families: 1,917
Train ratio: 98.6%
Statistics after balancing:
Number of train families: 1,878
Number of test families: 2,292
Train ratio: 87.5%

Split split_7
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,258
Number of test families: 1,912
Train ratio: 97.5%
Statistics after balancing:
Number of train families: 1,756
Number of test families: 2,414
Train ratio: 88.9%

Split split_8
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,258
Number of test families: 1,912
Train ratio: 98.1%
Statistics after balancing:
Number of train families: 1,910
Number of test families: 2,260
Train ratio: 88.9%

Last split
Splitting Rfamseq...
Statistics before balancing:
Number of train families: 2,276
Number of test families: 1,894
Train ratio: 99.5%
Statistics after balancing:
Number of train families: 2,011
Number of test families: 2,159
Train ratio: 84.7%
```
