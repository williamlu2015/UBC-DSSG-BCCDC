def deduplicate(_list):
    elements = set()

    lo = 0
    for hi, element in enumerate(_list):
        if element not in elements:
            _list[lo] = element
            lo += 1
            elements.add(element)

    while len(_list) > lo:
        _list.pop()
