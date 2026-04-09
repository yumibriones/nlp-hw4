import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import string
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typo
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    """
    All transformations are applied randomly with some fixed probability per transformation and per word.
    Edit the prob_dict to tune the probabilities of each transformation.
    For now the per word probabilities are fixed.
    Example original sentence: "That was so good!"
    """

    # Tune probabilities here!
    prob_dict = {
        "remove_spaces": 0.2,  # fairly common
        "replace_with_synonyms": 0.1,  # sometimes replaces with weird synonyms, so maybe not too high
        "replace_with_typos": 0.3, # pretty common but can really make text unreadable
        "random_capitalization": 0.5,  # this probably happens a lot
        "randomly_repeat_vowels": 0.3  # common in informal writing
    }

    detokenizer = TreebankWordDetokenizer()

    def _tokenize(text):
        # Fall back to whitespace tokenization if NLTK resources are unavailable.
        try:
            return word_tokenize(text)
        except LookupError:
            return text.split()

    def _is_word(token):
        return any(ch.isalpha() for ch in token)
    
    # Remove spaces
    # That was sogood!
    def _remove_spaces(text, sample_prob=0.1, word_prob=0.5):
        if random.random() >= sample_prob:
            return text
        words = text.split()
        if len(words) < 2:
            return text

        merged = [words[0]]
        for word in words[1:]:
            if random.random() < word_prob:
                merged[-1] += word
            else:
                merged.append(word)
        return " ".join(merged)
    
    # Replace words with synonyms
    # That was so excellent!
    def _replace_with_synonyms(text, sample_prob=0.1, word_prob=0.5):
        if random.random() >= sample_prob:
            return text

        words = _tokenize(text)
        new_words = []
        for word in words:
            if not _is_word(word) or random.random() >= word_prob:
                new_words.append(word)
                continue

            try:
                synsets = wordnet.synsets(word.lower())
            except LookupError:
                synsets = []

            candidates = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if (
                        candidate.lower() != word.lower()
                        and " " not in candidate
                        and candidate.isalpha()
                    ):
                        candidates.append(candidate)

            if candidates:
                replacement = random.choice(candidates)
                if word.istitle():
                    replacement = replacement.title()
                elif word.isupper():
                    replacement = replacement.upper()
                new_words.append(replacement)
            else:
                new_words.append(word)

        return detokenizer.detokenize(new_words)
    
    # Typos
    # That wsa so gopd!
    def _replace_with_typos(text, sample_prob=0.1, word_prob=0.5):
        if random.random() >= sample_prob:
            return text

        keyboard_neighbors = {
            "a": "sqwz",
            "e": "wrsd",
            "i": "uokj",
            "o": "ipkl",
            "s": "awedxz",
            "t": "rfgy",
            "n": "bhjm",
        }

        def _mutate_word(word):
            chars = list(word)
            alpha_indices = [i for i, ch in enumerate(chars) if ch.isalpha()]
            if not alpha_indices:
                return word

            idx = random.choice(alpha_indices)
            original = chars[idx]
            lower = original.lower()

            if lower in keyboard_neighbors and random.random() < 0.8:
                repl = random.choice(keyboard_neighbors[lower])
                chars[idx] = repl.upper() if original.isupper() else repl
            elif len(alpha_indices) > 1:
                swap_idx = idx + 1 if idx + 1 < len(chars) else idx - 1
                chars[idx], chars[swap_idx] = chars[swap_idx], chars[idx]

            return "".join(chars)

        tokens = _tokenize(text)
        new_tokens = []
        for token in tokens:
            if _is_word(token) and random.random() < word_prob:
                new_tokens.append(_mutate_word(token))
            else:
                new_tokens.append(token)

        return detokenizer.detokenize(new_tokens)
    
    # Randomly capitalize entire words
    # That was so GOOD!
    def _random_capitalization(text, sample_prob=0.1, word_prob=0.5):
        if random.random() >= sample_prob:
            return text

        words = _tokenize(text)
        new_words = []
        for word in words:
            if _is_word(word) and random.random() < word_prob:
                new_words.append(word.upper())
            else:
                new_words.append(word)
        return detokenizer.detokenize(new_words)

    # Randomly repeat vowels and punctuation
    # That was sooooo good!!!
    def _randomly_repeat_vowels(text, sample_prob=0.1, word_prob=0.5):
        if random.random() >= sample_prob:
            return text

        vowels = "aeiouAEIOU"
        tokens = _tokenize(text)
        new_tokens = []

        for token in tokens:
            if _is_word(token) and random.random() < word_prob:
                expanded = []
                for ch in token:
                    expanded.append(ch)
                    if ch in vowels and random.random() < 0.4:
                        expanded.extend(ch for _ in range(random.randint(1, 2)))
                new_tokens.append("".join(expanded))
            elif token in string.punctuation and random.random() < word_prob:
                new_tokens.append(token * random.randint(2, 3))
            else:
                new_tokens.append(token)

        return detokenizer.detokenize(new_tokens)

    # Wrapper function to apply all transformations
    def _apply_transformations(text, prob_dict):
        text = _remove_spaces(text, prob_dict["remove_spaces"])
        text = _replace_with_synonyms(text, prob_dict["replace_with_synonyms"])
        text = _replace_with_typos(text, prob_dict["replace_with_typos"])
        text = _random_capitalization(text, prob_dict["random_capitalization"])
        text = _randomly_repeat_vowels(text, prob_dict["randomly_repeat_vowels"])
        return text
    
    # Apply transformations to the example text
    example["text"] = _apply_transformations(example["text"], prob_dict)

    ##### YOUR CODE ENDS HERE ######

    return example
