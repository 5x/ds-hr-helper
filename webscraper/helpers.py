from .logger import logger


def url_filter(url, state, pattern):
    if not pattern.match(url):
        raise ValueError


def get_node_flat_string(tag, separator=' '):
    content = tag.get_text(separator=separator)
    content_parts = content.split()

    return separator.join(content_parts)


def append_to_file(filename, content, new_line=True):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(content)
        logger.info('Write line to file(%s).', filename)

        if new_line:
            file.write('\n')
