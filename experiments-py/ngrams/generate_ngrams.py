from typing import Iterable


def ngrams(n_word:str, lvl: int=0, q:int = 1) -> Iterable[str]:
    l = len(n_word)

    yield n_word

    if l > 2:

        lvl_next = lvl+1
        yield from ngrams(n_word[0: l-1], lvl_next, q)
        
        print(f"Right Node: {n_word[1:l]} q: {q}, q+2: {q+1} , lvl: {lvl}")
        if not (q+2 < lvl_next * 2):
            yield from ngrams(n_word[1: l], lvl_next, q+2)


def number_of_nodes(n_word:str):
    nodes = 2**len(n_word)
    dupes = 0
    
    """
    S=2n​⋅(a1​+an​)

Where:

    SS is the sum of the sequence.
    nn is the number of terms in the sequence.
    a1a1​ is the first term.
    anan​ is the last term.

In your sequence, a1=1a1​=1, n=4n=4, and an=3+1=4an​=3+1=4. Substituting these values into the formula:

S=42⋅(1+4)=2⋅5=10S=24​⋅(1+4)=2⋅5=10

So, the sum of the sequence [1, 1+1, 2+1, 3+1] is 10.
    """



if __name__ == "__main__":
    print(list(ngrams("hello")))