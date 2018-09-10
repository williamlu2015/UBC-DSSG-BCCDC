from wordcloud import WordCloud

from src.util.io import write_word_cloud


def word_cloud(
        elements, width=5120, height=2880, prefer_horizontal=1,
        min_font_size=90, max_words=100, stopwords=None,
        background_color="white", max_font_size=450, color_func=None,
        collocations=False, normalize_plurals=False, output_filename=None
):
    # stopwords=None causes the built-in STOPWORDS list to be used
    # collocations=False causes bigrams to be excluded
    # normalize_plurals=False prevents "previous" et al. from being truncated

    word_cloud_factory = WordCloud(
        width=width, height=height, prefer_horizontal=prefer_horizontal,
        min_font_size=min_font_size, max_words=max_words, stopwords=stopwords,
        background_color=background_color, max_font_size=max_font_size,
        color_func=color_func, collocations=collocations,
        normalize_plurals=normalize_plurals
    )

    if _are_documents(elements):
        text = " ".join(elements)
        word_cloud_instance = word_cloud_factory.generate(text)
    elif _are_word_frequencies(elements):
        word_cloud_instance = word_cloud_factory.fit_words(elements)
    else:
        raise ValueError

    image = word_cloud_instance.to_image()
    image.show()

    if output_filename is not None:
        write_word_cloud(output_filename, word_cloud_instance)
    return word_cloud_instance


def _are_documents(elements):
    return isinstance(elements, list)\
           and all(
        isinstance(item, str)
        for item in elements
    )


def _are_word_frequencies(elements):
    return isinstance(elements, dict) \
           and all(
        isinstance(key, str) and isinstance(value, float)
        for key, value in elements.items()
    )
