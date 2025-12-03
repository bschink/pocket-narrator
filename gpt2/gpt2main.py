from gpt2 import TextGenatate
from gpt2_trainer import Training



if __name__ == '__main__':

    tokenizer_cfg = "configs/tokenizers/tinystories_10k.yaml"
    train_cfg = "configs/training/training_medium.yaml"

    text_generate = TextGenatate(tokenizer_cfg)
    text_generate.main()

    training = Training(train_cfg)
    training.main()




    
    #textGenerate = TextGenatate()
    #textGenerate.main()

    #training = Training()
    #training.main()
