from utils.consts import *

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import IterableDataset
from torch.optim import Adam, SGD, AdamW
from utils.ml.optimizier import Sophia, Hutchinson


class Trainer:

    def __init__(self, model: Module,
                 epochs: int,
                 lr: float,
                 args: DataParams) -> None:
        self.model = model

        self.ep = epochs
        self.lr = lr

        self.bs = args.bs
        self.seq = args.seq

        self.args = args

        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        train_iter, val_iter, test_iter = WikiText2()

        self.train_data = self.data_process(train_iter)
        self.val_data = self.data_process(val_iter)
        self.test_data = self.data_process(test_iter)

        self.criterion = nn.CrossEntropyLoss()

        self.opt = Sophia(model.parameters(), Hutchinson(),
                          lr=lr, rho=.02,
                          weight_decay=0)  # Sophia-H optimizier
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2, gamma=0.9)

    def data_process(self, itter: IterableDataset) -> Tensor:
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in itter]
        data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        bs_seq = data.size(0) // self.bs
        data = data[:bs_seq * self.bs]

        return data.view(self.bs, bs_seq).t().contiguous().to(self.args.device)

    def get_batch(self, source: Tensor, i: int) -> Tuple[Tensor, Tensor]:

        seq = min(self.seq, len(source) - 1 - i)
        data = source[i:i + seq]
        target = source[i + 1:i + 1 + seq].reshape(-1)
        return data, target

    def train_step(self) -> None:

        batches = len(self.train_data) // self.seq
        log_int: int = 200
        loss_acc: float = 0.
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.seq)):
            x, y = self.get_batch(self.train_data, i)
            y_hat = self.model(x)

            loss = self.criterion(y_hat.view(-1, self.args.vocab), y)

            self.opt.zero_grad()
            loss.backward(retain_graph=true)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
            self.opt.step(loss)

            loss_acc += loss.item()

            if batch % log_int == 0 and batch > 0:
                self.lr = self.scheduler.get_last_lr()[0]
                cur_loss = loss_acc / log_int
                print(f'| {batch:5d}/{batches:5d} batches | '
                      f'lr {self.lr:02.10f} | '
                      f'loss {cur_loss:5.2f} ')

                loss_acc = 0.

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i in range(0, self.val_data.size(0) - 1, self.seq):
                data, targets = self.get_batch(self.val_data, i)
                seq_len = data.size(0)
                output = self.model(data)
                output_flat = output.view(-1, self.args.vocab)
                total_loss += seq_len * self.criterion(output_flat, targets).item()
        return total_loss / (len(self.val_data) - 1)

    def train(self) -> None:

        for _ in range(self.ep):
            epoch_start_time = time.time()

            self.train_step()

            val_loss = self.evaluate()
            val_ppl = np.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {_:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            self.scheduler.step()
