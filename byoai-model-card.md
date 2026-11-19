# Model card: Tiny GPT (character level)

## What it is

A decoder-only transformer ("GPT") written in PyTorch for the free course "Build Your Own AI and Automations Using Python". It has 118,353 parameters: 2 blocks, 4 attention heads, embedding size 64, a context of 128 characters and a vocabulary of 81 characters. It was trained for 5,000 steps (batch size 64) on a free Google Colab T4 GPU, which took about 94 seconds. Version 1, November 2026.

## Data

Three books from Project Gutenberg, taken from the NLTK Gutenberg corpus: *Alice's Adventures in Wonderland* (Carroll), *Persuasion* (Austen) and *Moby Dick* (Melville). The text was cleaned in Lesson 11 of the course. The last 10% of each book was kept for validation: 1,636,721 characters for training and 181,863 for validation.

## How well it works (measured in Lesson 14)

- Validation loss 1.489 (a bigram baseline gets 2.43).
- Perplexity 4.43 per character (the bigram gets 11.36).
- Training loss 1.445 against validation loss 1.489, so it is not overfit.
- About 0.89 of the words in a sample appear in the training text (0.99 for real held-out text), so roughly one word in nine is invented.
- It does not memorize its training text.

## Limits

- It writes old-book style text that mostly does not make sense.
- It has no facts. Given "The capital of France is " it never started with "Paris" in 20 tries, although "Paris" appears in the books. It hallucinates.
- It leans the way its three books lean. In the training text "he" appears 2,663 times and "she" 1,661 times.
- It only knows 81 characters, and it sees only the last 128 characters.
- It is not for real use: do not use it for facts, advice or decisions.

## Intended use

Learning: load it, change how it samples, read the code in `gpt_model.py`, and train your own.

## How to use

See `README.md`.
