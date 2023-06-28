from utils.consts import *
from utils.trainer import Trainer
from model.transformer import Transformer


def main() -> None:
    args = DataParams(
        dims=224,
        nheads=8,
        nblocks=6,
        scale=4,
        drop_rate=.1,
        batch_size=20,
        seq_len=35
    )
    model = Transformer(args).to(args.device)
    trainer = Trainer(model, epochs=10, lr=7e-3, args=args)

    trainer.train()


if __name__ == "__main__":
    main()
