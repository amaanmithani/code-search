class Stats:
    """Simple statistics helpers."""

    def mean(self, values):
        """Compute the arithmetic mean of a list of numbers and return it as a float."""
        return sum(values) / len(values)

    def maximum(self, values):
        """Return the largest number from a non-empty list of numeric values here."""
        return max(values)


def only_doc():
    """Placeholder function whose body is nothing but this rather long docstring."""
