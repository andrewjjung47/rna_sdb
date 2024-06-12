# Scripts for analysis

## Sequence similarities between different datasets

This analysis uses `cd-hit-est` to get number of unique sequences based on sequence similarity. It reports number of sequences in each dataset, number of unique sequences (i.e. cluster seeds of `cd-hit-est`), and number of sequences not similar to sequences in another dataset.

Example output:

```bash
python dataset_sequence_similarities.py 
2024-05-26 00:47:52,908 - Loading ArchiveII dataset...
2024-05-26 00:47:52,908 - Loading data from /home/andrewjung/rna_sdb/datasets/archiveII
2024-05-26 00:47:53,286 - Loaded 3966 examples.
Before filtering non-canonical bases: 3966
After filtering non-canonical bases: 3966
2024-05-26 00:47:53,288 - Loading bpRNA dataset...
2024-05-26 00:47:53,288 - Loading data from /home/andrewjung/rna_sdb/datasets/bprna
2024-05-26 00:47:54,209 - Loaded 13419 examples.
Before filtering non-canonical bases: 13419
After filtering non-canonical bases: 13345
2024-05-26 00:47:54,214 - Loading bpRNA-new dataset...
2024-05-26 00:47:54,214 - Loading data from /home/andrewjung/rna_sdb/datasets/bprna_new
2024-05-26 00:47:54,531 - Loaded 5401 examples.
Before filtering non-canonical bases: 5401
After filtering non-canonical bases: 5399
2024-05-26 00:47:54,533 - Loading RNAStrAlign dataset...
2024-05-26 00:47:54,533 - Loading data from rnastralign_train_no_redundant.seq.gz
2024-05-26 00:47:54,753 - Loading data from rnastralign_val_no_redundant.seq.gz
2024-05-26 00:47:54,774 - Loading data from rnastralign_test_no_redundant.seq.gz
2024-05-26 00:47:54,791 - Loaded 36259 examples.
2024-05-26 00:47:54,792 - Number of structures in each dataset:
2024-05-26 00:47:54,792 - ArchiveII: 3966
2024-05-26 00:47:54,792 - bpRNA: 13345
2024-05-26 00:47:54,792 - bpRNA-new: 5399
2024-05-26 00:47:54,792 - RNAStrAlign: 36259
2024-05-26 00:47:54,799 - Running sequence similarity analysis for ArchiveII dataset...
2024-05-26 00:47:54,799 - Number of sequences in ArchiveII dataset: 3966
Assigning cluster metadata to training sequences...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1231/1231 [00:00<00:00, 2086136.66it/s]
2024-05-26 00:47:56,726 - Number of unique sequences in "ArchiveII" dataset: 1231
Number of sequences in filtered training set: 3943 (number filtered 23)
2024-05-26 00:47:57,661 - Number of unique sequences in "ArchiveII" dataset that are not in "bpRNA-new" dataset: 3943
Number of sequences in filtered training set: 1033 (number filtered 2933)
2024-05-26 00:47:59,036 - Number of unique sequences in "ArchiveII" dataset that are not in "TR0" dataset: 1033
Number of sequences in filtered training set: 3352 (number filtered 614)
2024-05-26 00:47:59,758 - Number of unique sequences in "ArchiveII" dataset that are not in "TS0" dataset: 3352
Number of sequences in filtered training set: 1007 (number filtered 2959)
2024-05-26 00:48:11,568 - Number of unique sequences in "ArchiveII" dataset that are not in "RNAstralign-train" dataset: 1007
Number of sequences in filtered training set: 2178 (number filtered 1788)
2024-05-26 00:48:13,571 - Number of unique sequences in "ArchiveII" dataset that are not in "RNAstralign-test" dataset: 2178
2024-05-26 00:48:13,571 - Running sequence similarity analysis for bpRNA-new dataset...
2024-05-26 00:48:13,571 - Number of sequences in bpRNA-new dataset: 5399
Assigning cluster metadata to training sequences...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5273/5273 [00:00<00:00, 3206693.49it/s]
2024-05-26 00:48:16,108 - Number of unique sequences in "bpRNA-new" dataset: 5273
Number of sequences in filtered training set: 5359 (number filtered 40)
2024-05-26 00:48:17,960 - Number of unique sequences in "bpRNA-new" dataset that are not in "ArchiveII" dataset: 5359
Number of sequences in filtered training set: 5329 (number filtered 70)
2024-05-26 00:48:20,493 - Number of unique sequences in "bpRNA-new" dataset that are not in "TR0" dataset: 5329
Number of sequences in filtered training set: 5378 (number filtered 21)
2024-05-26 00:48:21,287 - Number of unique sequences in "bpRNA-new" dataset that are not in "TS0" dataset: 5378
Number of sequences in filtered training set: 5305 (number filtered 94)
2024-05-26 00:48:38,213 - Number of unique sequences in "bpRNA-new" dataset that are not in "RNAstralign-train" dataset: 5305
Number of sequences in filtered training set: 5372 (number filtered 27)
2024-05-26 00:48:40,821 - Number of unique sequences in "bpRNA-new" dataset that are not in "RNAstralign-test" dataset: 5372
2024-05-26 00:48:40,821 - Running sequence similarity analysis for TR0 dataset...
2024-05-26 00:48:40,821 - Number of sequences in TR0 dataset: 10756
Assigning cluster metadata to training sequences...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10633/10633 [00:00<00:00, 3251533.57it/s]
2024-05-26 00:48:45,859 - Number of unique sequences in "TR0" dataset: 10633
Number of sequences in filtered training set: 9815 (number filtered 941)
2024-05-26 00:48:48,589 - Number of unique sequences in "TR0" dataset that are not in "ArchiveII" dataset: 9815
Number of sequences in filtered training set: 10704 (number filtered 52)
2024-05-26 00:48:50,413 - Number of unique sequences in "TR0" dataset that are not in "bpRNA-new" dataset: 10704
Number of sequences in filtered training set: 10731 (number filtered 25)
2024-05-26 00:48:51,470 - Number of unique sequences in "TR0" dataset that are not in "TS0" dataset: 10731
Number of sequences in filtered training set: 9809 (number filtered 947)
2024-05-26 00:49:22,261 - Number of unique sequences in "TR0" dataset that are not in "RNAstralign-train" dataset: 9809
Number of sequences in filtered training set: 10438 (number filtered 318)
2024-05-26 00:49:26,466 - Number of unique sequences in "TR0" dataset that are not in "RNAstralign-test" dataset: 10438
2024-05-26 00:49:26,466 - Running sequence similarity analysis for TS0 dataset...
2024-05-26 00:49:26,466 - Number of sequences in TS0 dataset: 1298
Assigning cluster metadata to training sequences...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1295/1295 [00:00<00:00, 3439913.67it/s]
2024-05-26 00:49:28,715 - Number of unique sequences in "TS0" dataset: 1295
Number of sequences in filtered training set: 1162 (number filtered 136)
2024-05-26 00:49:29,612 - Number of unique sequences in "TS0" dataset that are not in "ArchiveII" dataset: 1162
Number of sequences in filtered training set: 1288 (number filtered 10)
2024-05-26 00:49:30,445 - Number of unique sequences in "TS0" dataset that are not in "bpRNA-new" dataset: 1288
Number of sequences in filtered training set: 1280 (number filtered 18)
2024-05-26 00:49:31,648 - Number of unique sequences in "TS0" dataset that are not in "TR0" dataset: 1280
Number of sequences in filtered training set: 1167 (number filtered 131)
2024-05-26 00:49:36,989 - Number of unique sequences in "TS0" dataset that are not in "RNAstralign-train" dataset: 1167
Number of sequences in filtered training set: 1250 (number filtered 48)
2024-05-26 00:49:38,062 - Number of unique sequences in "TS0" dataset that are not in "RNAstralign-test" dataset: 1250
2024-05-26 00:49:38,062 - Running sequence similarity analysis for RNAstralign-train dataset...
2024-05-26 00:49:38,062 - Number of sequences in RNAstralign-train dataset: 29719
Assigning cluster metadata to training sequences...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2954/2954 [00:00<00:00, 837386.73it/s]
2024-05-26 00:49:43,875 - Number of unique sequences in "RNAstralign-train" dataset: 2954
Number of sequences in filtered training set: 14281 (number filtered 15438)
2024-05-26 00:49:48,999 - Number of unique sequences in "RNAstralign-train" dataset that are not in "ArchiveII" dataset: 14281
Number of sequences in filtered training set: 29714 (number filtered 5)
2024-05-26 00:49:52,151 - Number of unique sequences in "RNAstralign-train" dataset that are not in "bpRNA-new" dataset: 29714
Number of sequences in filtered training set: 19349 (number filtered 10370)
2024-05-26 00:49:58,117 - Number of unique sequences in "RNAstralign-train" dataset that are not in "TR0" dataset: 19349
Number of sequences in filtered training set: 26147 (number filtered 3572)
2024-05-26 00:49:59,879 - Number of unique sequences in "RNAstralign-train" dataset that are not in "TS0" dataset: 26147
Number of sequences in filtered training set: 6115 (number filtered 23604)
2024-05-26 00:50:09,391 - Number of unique sequences in "RNAstralign-train" dataset that are not in "RNAstralign-test" dataset: 6115
2024-05-26 00:50:09,392 - Running sequence similarity analysis for RNAstralign-test dataset...
2024-05-26 00:50:09,392 - Number of sequences in RNAstralign-test dataset: 2825
Assigning cluster metadata to training sequences...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 728/728 [00:00<00:00, 1481539.70it/s]
2024-05-26 00:50:11,062 - Number of unique sequences in "RNAstralign-test" dataset: 728
Number of sequences in filtered training set: 1351 (number filtered 1474)
2024-05-26 00:50:12,145 - Number of unique sequences in "RNAstralign-test" dataset that are not in "ArchiveII" dataset: 1351
Number of sequences in filtered training set: 2825 (number filtered 0)
2024-05-26 00:50:13,009 - Number of unique sequences in "RNAstralign-test" dataset that are not in "bpRNA-new" dataset: 2825
Number of sequences in filtered training set: 1939 (number filtered 886)
2024-05-26 00:50:14,148 - Number of unique sequences in "RNAstralign-test" dataset that are not in "TR0" dataset: 1939
Number of sequences in filtered training set: 2540 (number filtered 285)
2024-05-26 00:50:14,892 - Number of unique sequences in "RNAstralign-test" dataset that are not in "TS0" dataset: 2540
Number of sequences in filtered training set: 186 (number filtered 2639)
2024-05-26 00:50:23,632 - Number of unique sequences in "RNAstralign-test" dataset that are not in "RNAstralign-train" dataset: 186
2024-05-26 00:50:23,632 - Summary of sequence similarity analysis:
              dataset   compared_dataset  num_unique_sequences
0           ArchiveII          ArchiveII                  1231
1           ArchiveII          bpRNA-new                  3943
2           ArchiveII                TR0                  1033
3           ArchiveII                TS0                  3352
4           ArchiveII  RNAstralign-train                  1007
5           ArchiveII   RNAstralign-test                  2178
6           bpRNA-new          bpRNA-new                  5273
7           bpRNA-new          ArchiveII                  5359
8           bpRNA-new                TR0                  5329
9           bpRNA-new                TS0                  5378
10          bpRNA-new  RNAstralign-train                  5305
11          bpRNA-new   RNAstralign-test                  5372
12                TR0                TR0                 10633
13                TR0          ArchiveII                  9815
14                TR0          bpRNA-new                 10704
15                TR0                TS0                 10731
16                TR0  RNAstralign-train                  9809
17                TR0   RNAstralign-test                 10438
18                TS0                TS0                  1295
19                TS0          ArchiveII                  1162
20                TS0          bpRNA-new                  1288
21                TS0                TR0                  1280
22                TS0  RNAstralign-train                  1167
23                TS0   RNAstralign-test                  1250
24  RNAstralign-train  RNAstralign-train                  2954
25  RNAstralign-train          ArchiveII                 14281
26  RNAstralign-train          bpRNA-new                 29714
27  RNAstralign-train                TR0                 19349
28  RNAstralign-train                TS0                 26147
29  RNAstralign-train   RNAstralign-test                  6115
30   RNAstralign-test   RNAstralign-test                   728
31   RNAstralign-test          ArchiveII                  1351
32   RNAstralign-test          bpRNA-new                  2825
33   RNAstralign-test                TR0                  1939
34   RNAstralign-test                TS0                  2540
35   RNAstralign-test  RNAstralign-train                   186
```