from create_model import create_model
from model import test_model

def main():
    """
    model version 1: input_shape = (250, 250, 3)
    model version 1: input_shape = (500, 500, 3)
    """
    input_shape = (500, 500, 3)
    create_model(input_shape)
    #test_model(input_shape)

if __name__ == "__main__":
    main()