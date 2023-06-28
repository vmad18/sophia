from utils.ml.layers import *
from utils.consts import *


class Encoder(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.mha = MultiHeadAttention(args)
        self.ffn = FFN(args)

        self.l1 = nn.LayerNorm(args.dims, eps=args.eps)
        self.l2 = nn.LayerNorm(args.dims, eps=args.eps)

    def forward(self, x: Tensor, mask: Tensor = null) -> Tensor:
        x = self.l1(self.mha(x, mask) + x)
        return self.l2(self.ffn(x) + x)


class Decoder(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.mha = CrossAttention(args)
        self.m_mha = MultiHeadAttention(args)
        self.ffn = FFN(args)

        self.l1 = nn.LayerNorm(args.dims, eps=args.eps)
        self.l2 = nn.LayerNorm(args.dims, eps=args.eps)
        self.l3 = nn.LayerNorm(args.dims, eps=args.eps)

    def forward(self, x: Tensor, enc: Tensor, mask: Tensor = null) -> Tensor:
        x = self.l1(self.m_mha(x, mask) + x)
        x = self.l2(self.mha(x, enc) + x)
        return self.l3(self.ffn(x) + x)


class Transformer(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.args = args

        self.pos_enc = PositionalEncoding(args)
        self.embed = nn.Embedding(args.vocab, args.dims)

        self.encoder = nn.ModuleList([Encoder(args) for _ in range(args.blcks)])
        # self.decoder = nn.ModuleList([Decoder(args) for _ in range(args.blcks)])

        self.classifier = nn.Linear(args.dims, args.vocab)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp: Tensor, target: Tensor = null) -> Tensor:

        x = self.pos_enc(self.embed(inp) * torch.sqrt(torch.tensor([self.args.dims], device=self.args.device)))

        for blk in self.encoder: x = blk(x)
        ctx = x

        if target != null:

            x = self.pos_enc(self.embed(target))

            T = inp.shape[1]

            mask = (torch.triu(torch.ones(T, T)) - torch.diag(torch.ones(T))) * -1e9

            for blk in self.decoder: x = blk(x, ctx, mask)

        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = Transformer(args=model_args)
