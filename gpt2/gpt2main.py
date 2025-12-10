import wandb
from datetime import datetime

from gpt2 import TextGenatate
from gpt2_trainer import Training



if __name__ == '__main__':

    tokenizer_cfg = "configs/tokenizers/tinystories_1M.yaml"
    train_cfg = "configs/training/train_tinystories_1M.yaml"

    # Initialize wandb BEFORE training starts
    wandb.init(
        entity="once-upon-a-prompt",
        project="gpt2",
        name=f"gpt2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    text_generate = TextGenatate(tokenizer_cfg)
    text_generate.main()

    training = Training(train_cfg)
    training.main()

    wandb.finish()
