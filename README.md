# Master_Thesis
All code for my master thesis on computational approaches to the study of systematicity and iconicity in language acquisition

The map childes_corpus_reader contains all code required to process the CHILDES corpus and extract from it 3 lists of lists for both child directed speech and child produced speech. The first list contains all tokens from the corpus, the second all corresponding lemmas, and the third all corresponding POS-tags. The corpus can be split according to the ages of the target children. The code was based on previously written code by Dr. Giovanni Cassani and Dr. Robert Grimm.

The map celex_reader contains all code required to process two of the obtained lists from the childes_corpus_reader code (the token list and the POS-tag list) to obtain form-embeddings for all tokens in the CHILDES corpus. To use it, you will need a copy of the CELEX database and a file containing a list of POS mappings, which can map the tags from the CHILDES corpus to the CELEX corpus. The code was almost entirely made by Dr. Giovanni Cassani, with some minor tweaks.

The map NDL_semantic_vectors contains all code required to make semantic vectors with Naive Discriminative Learning. You will need the list of tokens extracted from the CHILDES corpus with the childes_corpus_reader code to use as cues. You can then define your desired outcomes, or use this same token list as outcomes if you want to learn the direct mapping of tokens. The code was almost entirely made by Dr. Giovanni Cassani, with some minor tweaks.

The file helper_functions.py contains various functions that were used throughout the project for making semantic vectors, form vectors, obtaining wordcounts and map words to when they were first produced
