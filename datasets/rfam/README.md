# Rfam: The RNA families database

The Rfam database is a collection of RNA sequence families of structural RNAs including non-coding RNA genes as well as cis-regulatory elements. Each family is represented by a multiple sequence alignment and a covariance model (CM) [1].

For RNA-SDB, Rfam version 14.10 is used (the most up-to-date at the time of publication), and relevant files are downloaded from Rfam's [FTP site](https://ftp.ebi.ac.uk/pub/databases/Rfam/14.10/).

## Files used for RNA-SDB

Here, we describe what the following Rfam files are and how they are used to construct RNA-SDB.

### Rfam.full_region.gz

`Rfam.full_region.gz` contains list of sequences which make up the full family membership for each family. The file is tab separated, and fields are as follows:

1. Rfam accession (e.g. `RF00001`)
2. The sequence accession and version number (e.g. `EU093378.1`)
3. Start coordinate of match on sequence
4. End coordinate of match on sequence
5. Bitscore
6. E-value
7. CM start position
8. CM end position
9. Binary flag whether a match is a truncated match to CM
10. Type of sequence, either seed or full

### Rfam.seed.gz

`Rfam.seed.gz` contains seed alignments for all Rfam families. The file is concatenation of STOCKHOLM format files for each Rfam families.

References:

- [1] <https://docs.rfam.org/en/latest/about-rfam.html>
