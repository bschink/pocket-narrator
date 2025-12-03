import wandb
import datetime

from gpt2 import TextGenatate
from gpt2_trainer import Training



if __name__ == '__main__':

    tokenizer_cfg = "configs/tokenizers/tinystories_10k.yaml"
    train_cfg = "configs/training/training_medium.yaml"

    text_generate = TextGenatate(tokenizer_cfg)
    text_generate.main()

    training = Training(train_cfg)
    training.main()

    wandb.init(
        entity="once-upon-a-prompt",
        project="gpt2",
        run_name=f"gpt2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )




    
    #textGenerate = TextGenatate()
    #textGenerate.main()

    #training = Training()
    #training.main()
