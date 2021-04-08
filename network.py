import torch
import torch.nn as nn
from model import Model
from bonito import model as bmodel
import bonito
import toml
import decoder.decoder as d
import numpy as np
from itertools import product

class SignalEncoder(nn.Module):
    def __init__(self, encoder):
        super(SignalEncoder, self).__init__()
        
        self.e = encoder
        
    def run_m(self, m, mask, masksm):
        def run(x):
            _x = x
            for layer in m.conv:
                _x = layer(_x)
                #print(_x.shape, mask.shape)
                if _x.shape[2] == mask.shape[2]:
                    _x = _x * mask
                else:
                    _x = _x * masksm
            if m.use_res:
                _x += m.residual(x) * mask
            return m.activation(_x) * mask
        return run
    
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False): 
            #print("start size", x.shape)
            shape = x.shape
            x = x.permute((0,2,1))
            x_evs, lens, moves, _ = self.e.encoder[0](x)
            
            mask = torch.ones((x_evs.shape[0], 1, x_evs.shape[2])).cuda()
            masksm = torch.ones((x_evs.shape[0], 1, x_evs.shape[2] // 3)).cuda()
            for i, l in enumerate(lens):
                mask[i,:,l:] = 0
                masksm[i,:,(l+2)//3:] = 0
        
            #print(x_evs.shape)
            for block in self.e.encoder[1:]:
                #x_evs = checkpoint(self.run_m(block, mask), x_evs)
                x_evs = self.run_m(block, mask, masksm)(x_evs)
                #print(x_evs.shape, lens, mask.shape)
                #x_evs = x_evs * mask
                
            #print("after c", x.shape)
            x_evs = x_evs.permute((0,2,1))
            #print(b - a, c - b, d - c, e - d, x_evs.cpu().shape, samples.shape)
            lens = torch.IntTensor(lens).cuda()
            return x_evs, lens#, x.shape[2] / moves.sum(dim=1)

    
class BaseEncoder(nn.Module):
    def __init__(self, ks=48*5+5, k=6):
        super(BaseEncoder, self).__init__()

        contexts = []
        for i in range(k+1):
            contexts += list(product([0,1,2,3,4], repeat=i))

        self.context_dict = {x: i for i, x in enumerate(contexts)}

        self.emb = nn.Embedding(len(self.context_dict), ks)
        self.emb.weight.data *= 0.03

        self.base_emb = nn.Parameter((torch.rand(ks) - 0.5) / 3)
        self.k = k


    def forward(self, x):
        indices = []
        x = x.detach().cpu().numpy()
        for row in x:
            row_ids = [self.context_dict[tuple(row[max(0, i-self.k):i])] for i in range(len(row)+1)]
            indices.append(torch.LongTensor(row_ids))
        inds = torch.stack(indices).cuda()
        return self.emb(inds) + self.base_emb
    
class Joiner(nn.Module):
    def __init__(self, sks=96, bks=32, hidden=96):
        super(Joiner, self).__init__()
        #self.sh = nn.Linear(sks, hidden)
        #self.bh = nn.Linear(bks, hidden)
        #self.out = nn.Linear(hidden, 5)
        
    
    def proc_chunk(self):
        def go(s, b):
            t, x_h = s.size()
            u, y_h = b.size()
            s = s.unsqueeze(dim=1).expand(torch.Size([t, u, x_h]))
            b = b.unsqueeze(dim=0).expand(torch.Size([t, u, y_h]))

            #print(s.shape, b.shape)

            z = s*torch.nn.functional.relu(b)
            z = self.out(z)
            z = F.log_softmax(z, dim=-1)
            return z
        return go
    
    def forward(self, s, b):
        s = self.sh(s)
        b = self.bh(b)
        
        sparts = torch.unbind(s, 0)
        bparts = torch.unbind(b, 0)
        
        to_stack = []
#        print(s.shape, b.shape)
        step = 1
        for i in range(0, len(sparts), step):
            to_stack.append(checkpoint(self.proc_chunk(), sparts[i], bparts[i]))
#            print(to_stack[-1].shape)
        ret = torch.stack(to_stack, dim=0)
        #print(ret.shape, s.shape, b.shape)
        return ret
        
    
    def simple_sp(self, s, b):
        b = self.bh(b)
        z = torch.tanh(s+b)
        z = self.out(z)
        z = F.log_softmax(z, dim=-1)
        return z

        
class Net(nn.Module):
    def __init__(self, s, b, j):
        super(Net, self).__init__()
        self.s = s
        self.b = b
        self.j = j
        
    def forward(self, x, y):
        s = self.s(x)
        b = self.b(y)
        return self.j(s, b), s, b

def create_network():
    cfgx = toml.load(bonito.__path__[0] + "/models/configs/dna_r9.4.1.toml")
    cfgx["block"][0]["type"] = "dynpool"
    cfgx["block"][0]["norm"] = 3
    cfgx["block"][0]["predictor_size"] = 128

    for b in cfgx['block']:
        if b["residual"]:
            b["type"] = 'BlockX'
            b["pool"] = 3
            b["inner_size"] = int(b["filters"]*2)
            kern = max(3, b["kernel"][0] // 3)
            kern = kern // 2 * 2 + 1
            b["kernel"] = [kern]

    bmodelx = Model(cfgx)
        
    
    sencoder = SignalEncoder(bmodelx.encoder)
    be = BaseEncoder()

    joiner = Joiner()

    model = Net(sencoder, be, joiner)
    
    model.load_state_dict(torch.load("weights/network.pth"))

    model.eval()
    model.cpu()
    model.s.cuda()
    return model.s

def create_decoder():
    cfgx = toml.load(bonito.__path__[0] + "/models/configs/dna_r9.4.1.toml")
    cfgx["block"][0]["type"] = "dynpool"
    cfgx["block"][0]["norm"] = 3
    cfgx["block"][0]["predictor_size"] = 128

    for b in cfgx['block']:
        if b["residual"]:
            b["type"] = 'BlockX'
            b["pool"] = 3
            b["inner_size"] = int(b["filters"]*2)
            kern = max(3, b["kernel"][0] // 3)
            kern = kern // 2 * 2 + 1
            b["kernel"] = [kern]

    bmodelx = Model(cfgx)
        
    
    sencoder = SignalEncoder(bmodelx.encoder)
    be = BaseEncoder()

    joiner = Joiner()

    model = Net(sencoder, be, joiner)
    
    model.load_state_dict(torch.load("weights/network.pth"))

    model.eval()
    model.cpu()
    
    torch.set_grad_enabled(False)

    small_tables = []

    for i in range(7):
        contexts = sorted(list(product([1,2,3,4], repeat=i)))
        to_tab = []
        for c in contexts:
            to_tab.append(np.pad(
                (model.b.emb.weight[model.b.context_dict[c]] + model.b.base_emb).numpy(),
                (0,11), constant_values=0))


        tab = np.stack(to_tab)
        print(i, tab.shape)
        small_tables.append(tab)

    decoder = d.DecoderTab(small_tables[0],
                           small_tables[1],
                           small_tables[2],
                           small_tables[3],
                           small_tables[4],
                           small_tables[5],
                           small_tables[6], 
    )

    return decoder
