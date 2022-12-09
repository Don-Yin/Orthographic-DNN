from utils.data_generate.main import init_create_prime_data, init_create_train_data

if __name__ == "__main__":
    """
    -------------------------------------------------------------------
    init_create_train_data: generates images of 1000 words with each
    word present in multiple images. Each image varies in letter 
    location, rotation, font as well as size. The size of the variation
    can be modified at utils.data_generate.main
    params: 
        dummy: if True make a smaller dataset
        random: if True use random strings instead of words
    -------------------------------------------------------------------
    init_create_prime_data: generate images of the 420 words from the
    Form Priming Project (FPP) across 28 priming conditions (see Adelman, 
    2014). The images are free of variations, all words are presented at
    the center.
    params:
        position_correction: if True, the position of the strings that
        are shorter than the target string is corrected (e.g., "SUB3"
        type will be moved to either the left or right). Defaults to 
        False.
    -------------------------------------------------------------------
    """
    init_create_train_data(dummy=False, random=False)
    init_create_prime_data(position_correction=False)
