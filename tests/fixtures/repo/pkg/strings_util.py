def reverse_words(sentence):
    """Reverse the order of the words in a sentence while keeping each word intact."""
    return " ".join(reversed(sentence.split()))


def count_vowels(text):
    """Count how many vowels appear in the given text, ignoring letter case entirely."""
    return sum(ch in "aeiou" for ch in text.lower())


def short(x):
    """Too short to count."""
    return x
