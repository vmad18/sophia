from utils.consts import *


class ScaledDotProduct(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.args = args

        self.drop = nn.Dropout(args.dr)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = null) -> Tensor:
        attn = (q @ k.transpose(-2, -1)) / torch.tensor([self.args.sd], device=self.args.device).sqrt()
        if mask != null: attn += mask
        attn = F.softmax(attn, dim=-1)

        cross = self.drop(attn @ v).permute(0, 2, 1, 3)
        cross = cross.reshape(cross.shape[0], cross.shape[1], self.args.dims)
        return cross


"""
Linear "Attention" Mechanism
Time Complexity: O(TD)
"""


class HydraAttention(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.nheads = args.dims

        self.to_qkv = nn.Linear(args.dims, 3 * args.dims)
        self.proj = nn.Linear(args.dims, args.dims)

        self.drop = nn.Dropout(args.dr)

    def forward(self, x: Tensor, mask: Tensor = null) -> Tensor:
        b, s, d = x.shape

        q, k, v = self.to_qkv(x).view(b, s, self.nheads, 3).permute(0, 2, 1, 3).chunk(3, dim=-1)

        q = q / q.norm(dim=-1, keepdim=true)
        k = k / k.norm(dim=-1, keepdim=true)

        aggregated = (k * v).sum(dim=-2, keepdims=true)
        gated = self.drop(q * aggregated)

        gated = gated.permute(0, 2, 1, 3).view(b, s, d)

        return self.proj(gated)


class MultiHeadAttention(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.args = args

        self.base = ScaledDotProduct(args)

        self.to_qkv = nn.Linear(args.dims, 3 * args.dims)
        self.proj = nn.Linear(args.dims, args.dims)

        # self.proj.weight.data.uniform_(-.1, .1)
        # self.to_qkv.weight.data.uniform_(-.1, .1)

        # nn.init.normal_(self.to_qkv.weight, mean=0., std=np.sqrt(2 / (args.dims + args.dims)))

    def forward(self, x: Tensor, mask: Tensor = null) -> Tensor:

        if not self.args.bf: x = x.transpose(0, 1)

        b, s, d = x.shape

        q, k, v = self.to_qkv(x).view(b, s, self.args.h, 3 * self.args.sd).permute(0, 2, 1, 3).chunk(3, -1)

        attn = self.base(q, k, v, mask)

        if not self.args.bf: attn = attn.transpose(0, 1)

        return self.proj(attn)


class CrossAttention(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.args = args

        self.base = ScaledDotProduct(args)

        self.to_qk = nn.Linear(args.dims, 2 * args.dims)
        self.to_v = nn.Linear(args.dims, args.dims)

        self.proj = nn.Linear(args.dims, args.dims)

    def forward(self, x: Tensor, ctx: Tensor, mask: Tensor = null) -> Tensor:

        if not self.args.bf: x = x.transpose(0, 1)

        b, s, d = x.shape

        q, k = self.to_qk(ctx).view(b, s, self.args.h, 2 * self.args.sd).permute(0, 2, 1, 3).chunk(2, -1)
        v = self.to_v(x).view(b, s, self.args.h, self.args.sd).permute(0, 2, 1, 3)

        attn = self.base(q, k, v, mask)

        if not self.args.bf: attn = attn.transpose(0, 1)

        return self.proj(attn)


class FFN(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.proj = nn.Linear(args.dims, args.scale * args.dims)
        self.proj_back = nn.Linear(args.scale * args.dims, args.dims)

        self.drop = nn.Dropout(args.dr)

        self.nl = args.nl
        self.proj.weight.data.uniform_(-.1, .1)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj_back(self.nl(self.proj(x))))


class PositionalEncoding(Module):

    def __init__(self, args: DataParams):
        super().__init__()

        self.args = args

        pos = torch.arange(0, args.max_toks).unsqueeze(1).to(args.device)
        vals = torch.exp(torch.arange(0, args.dims, 2) / args.dims * -np.log(1e4)).to(args.device)

        self.encodes = torch.zeros(args.max_toks, 1, args.dims).to(args.device)  # (seq, batch, dims)
        self.encodes[:, 0, 0::2] = (pos * vals).sin()
        self.encodes[:, 0, 1::2] = (pos * vals).cos()

        self.drop = nn.Dropout(args.dr)

    def forward(self, x: Tensor) -> Tensor:
        s, b, c = x.shape
        return self.drop(x + self.encodes[:s])


def main(x: Tensor, mask: Tensor = null) -> Tensor:
    proj = nn.Linear(x.shape[-1], model_args.dims)
    pos_enc = PositionalEncoding(model_args)
    mha = MultiHeadAttention(model_args)
    return mha(pos_enc(proj(x)), mask)


if __name__ == "__main__":

    tnsr = torch.randn(3, 10, 200)
    hydra = HydraAttention(model_args)
