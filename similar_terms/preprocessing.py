def preprocessing_word(word: str,
                       b_lower=True,
                       b_join: bool = True
                       ):
    """Do basic processing to be able to join identical words.
    TODO: update with Francois' methods

    Args:
        word:
        b_join: Boolean to replace spaces with '_' or not
    Returns:

    """

    word_pre = word[:]  # copy

    word_pre = word_pre.strip()

    word_pre = word_pre.replace('  ', ' ')  # Remove double spaces
    if b_lower:
        word_pre = lower_case(word_pre)
    if b_join:
        word_pre = word_pre.replace(' ', '_')
    word_pre = word_pre.replace('__', '_')  # Remove double underscores

    return word_pre


def lower_case(s: str) -> str:
    return s.lower()


def strip(s: str) -> str:
    return s.strip()


def single_space(s: str) -> str:
    return s.replace('  ', ' ')


def join_words(s: str, join='_'):
    return s.replace(' ', join)


def single_underscore(s: str) -> str:
    return s.replace('__', '_')
