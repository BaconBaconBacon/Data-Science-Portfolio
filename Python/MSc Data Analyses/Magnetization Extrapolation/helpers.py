def flatten_list(lst: list) -> list:
    """Flatten a nested list into a single-level list.

    Args:
        lst: Nested list to flatten.

    Returns:
        Flattened list containing all elements.
    """
    return [item for sublist in lst for item in sublist]
