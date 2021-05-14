import torch
import torch.nn as nn
from model import Model
from bonito import model as bmodel
import bonito
import toml
import decoder.decoder as d
import numpy as np
from itertools import product
from datetime import datetime

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def batchify(x, size, padding, max_batch_size=50):
    seq_size = x.shape[1]
    start_pos = 0

    pads = []
    chunks = []

    while start_pos < seq_size:
        if start_pos + size > seq_size:
            start = seq_size - size
            end = seq_size
            end_pad = size
            start_pad = (start_pos + padding) - start
        else:
            start = start_pos
            end = start_pos + size
            start_pad = 0 if start == 0 else padding
            end_pad = size - padding

        chunks.append(x[0,start:end,:])
        pads.append((start_pad, end_pad))

        start_pos += size - 2*padding

    out_chunks = []
    out_pads = []
    for i in range(0, len(chunks), max_batch_size):
        out_chunks.append(torch.stack(chunks[i:i+max_batch_size], dim=0))
        out_pads.append(pads[i:i+max_batch_size])

    return out_chunks, out_pads
            


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

    def do_pool(self, first_chunks, first_pads):
        enc = self.e.encoder[0]

        all_features = []
        all_jumps = []
        for fc, fp in zip(first_chunks, first_pads):
            fc = fc.permute(0,2,1)
            _x = fc
            for layer in enc.conv:
                _x = layer(_x)
            _x = enc.activation(_x)
            jumps_mat = torch.sigmoid(enc.predictor(torch.cat([fc, fc*fc], dim=1)))
            for rf, rj, (ps, pe) in zip(_x.permute(0,2,1), jumps_mat.permute(0,2,1), fp):
                all_features.append(rf[ps:pe])
                all_jumps.append(rj[ps:pe])

        with torch.cuda.amp.autocast(enabled=False): 
            features = torch.cat(all_features, dim=0)
            jumps = torch.cat(all_jumps).to(torch.float32)

            weights = jumps[:,0]
            moves = jumps[:,1] * enc.norm_mean

            pooled = enc.row_pool(features, moves, weights).unsqueeze(0)


        return pooled

        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True): 
            print("start size", x.shape)
            first_chunks, first_pads = batchify(x, 2048, 50, 32)
            x_evs = self.do_pool(first_chunks, first_pads)

            second_chunks, second_pads = batchify(x_evs, 512*3, 20, 32)

            out = []
            for sc, sp in zip(second_chunks, second_pads):
                x_evs = sc.permute(0, 2, 1)
                for block in self.e.encoder[1:]:
                    x_evs = block(x_evs)
                x_evs = x_evs.permute((0,2,1)).float().cpu()

                for r, (ps, pe) in zip(x_evs, sp):
                    out.append(r[ps:pe])
            res = torch.cat(out, dim=0).unsqueeze(0)
            return res

    
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

    torch.quantization.fuse_modules(model, [
        ('s.e.encoder.1.residual.0.conv', 's.e.encoder.1.residual.1'),
        ('s.e.encoder.1.conv.2', 's.e.encoder.1.conv.3'),

        ('s.e.encoder.2.residual.0.conv', 's.e.encoder.2.residual.1'),
        ('s.e.encoder.2.conv.2', 's.e.encoder.2.conv.3'),
        ('s.e.encoder.2.conv.6.pointwise', 's.e.encoder.2.conv.7'),
        ('s.e.encoder.2.conv.10.pointwise', 's.e.encoder.2.conv.11'),
        ('s.e.encoder.2.conv.14.pointwise', 's.e.encoder.2.conv.15'),
        ('s.e.encoder.2.conv.18.pointwise', 's.e.encoder.2.conv.19'),
        ('s.e.encoder.2.conv.22.pointwise', 's.e.encoder.2.conv.23'),
        
        ('s.e.encoder.3.residual.0.conv', 's.e.encoder.3.residual.1'),
        ('s.e.encoder.3.conv.2', 's.e.encoder.3.conv.3'),
        ('s.e.encoder.3.conv.6.pointwise', 's.e.encoder.3.conv.7'),
        ('s.e.encoder.3.conv.10.pointwise', 's.e.encoder.3.conv.11'),
        
        ('s.e.encoder.4.residual.0.conv', 's.e.encoder.4.residual.1'),
        ('s.e.encoder.4.conv.2', 's.e.encoder.4.conv.3'),
        ('s.e.encoder.4.conv.6.pointwise', 's.e.encoder.4.conv.7'),
        ('s.e.encoder.4.conv.10.pointwise', 's.e.encoder.4.conv.11'),
        ('s.e.encoder.4.conv.14.pointwise', 's.e.encoder.4.conv.15'),
        ('s.e.encoder.4.conv.18.pointwise', 's.e.encoder.4.conv.19'),
        ('s.e.encoder.4.conv.22.pointwise', 's.e.encoder.4.conv.23'),
        ('s.e.encoder.4.conv.26.pointwise', 's.e.encoder.4.conv.27'),
        ('s.e.encoder.4.conv.30.pointwise', 's.e.encoder.4.conv.31'),
        
        ('s.e.encoder.5.residual.0.conv', 's.e.encoder.5.residual.1'),
        ('s.e.encoder.5.conv.2', 's.e.encoder.5.conv.3'),
        ('s.e.encoder.5.conv.6.pointwise', 's.e.encoder.5.conv.7'),
        ('s.e.encoder.5.conv.10.pointwise', 's.e.encoder.5.conv.11'),
        ('s.e.encoder.5.conv.14.pointwise', 's.e.encoder.5.conv.15'),
        ('s.e.encoder.5.conv.18.pointwise', 's.e.encoder.5.conv.19'),
        
        ('s.e.encoder.6.conv.0.pointwise', 's.e.encoder.6.conv.1'),
        ('s.e.encoder.7.conv.0.conv', 's.e.encoder.7.conv.1'),

        ('s.e.encoder.0.conv.0.conv', 's.e.encoder.0.conv.1'),
        ('s.e.encoder.0.predictor.0', 's.e.encoder.0.predictor.1'),
        ('s.e.encoder.0.predictor.3', 's.e.encoder.0.predictor.4'),
        
    ], inplace=True)

    print(model.s.e.encoder)

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
