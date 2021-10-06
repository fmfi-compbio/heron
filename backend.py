import numpy as np
from datetime import datetime
import torch.multiprocessing as mp
import queue
from network import create_network, create_decoder
from scipy.signal import find_peaks
import torch

def _slice(raw_signal, start, end):
    pad_start = max(0, -start)
    pad_end = min(max(0, end - len(raw_signal)), end - start)
    return (
        np.pad(
            raw_signal[max(0, start) : min(end, len(raw_signal))],
            (pad_start, pad_end),
            constant_values=(0, 0),
        ),
        pad_start,
        pad_end,
    )


def med_mad(x, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def norm_by_noisiest_section(signal, samples=100, threshold=6.0):
    """
    Normalise using the medmad from the longest continuous region where the
    noise is above some threshold relative to the std of the full signal.
    """
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        med, mad = med_mad(signal[info['left_bases'][widest]: info['right_bases'][widest]])
    else:
        med, mad = med_mad(signal)
    return (signal - med) / mad

def rescale_signal(signal):
    signal = signal.astype(np.float32)
#    med, mad = med_mad(signal)
#    signal -= med
#    signal /= mad
    signal = norm_by_noisiest_section(signal) 
    return np.clip(signal, -3, 3)

def signal_to_chunks(raw_signal, metadata, s_len, pad):
    raw_signal = rescale_signal(raw_signal)
    pos = 0
    while pos < len(raw_signal):
        # assemble batch
        signal, pad_start, pad_end = _slice(
            raw_signal, pos - pad, pos - pad + s_len
        )
        crop = slice(max(pad, pad_start) // 3, -max(pad, pad_end) // 3)
        bound = metadata if pos == 0 else None
        pos += s_len - 2 * pad

        yield (bound, signal, crop)

def caller_process(model_file, qin, qout):
    caller_network = create_network(model_file)
    tc = 0
    with torch.no_grad():
        while True:
            item = qin.get()
            if item == "wait":
                qout.put("wait")
                continue
            if item is None:
                qout.put(None)
                break 
            signal = item[1]
            tc += 1
            output = caller_network(torch.tensor(signal).cuda())
            qout.put((item[0], output, item[2]))
    print("caller done", tc)

def finalizer_process(model_file, qin, qout, beam_size, beam_cut_threshold):
    decoder = create_decoder(model_file) 

    item = "wait"
    tc = 0
    while item == "wait":
        cur_name = ""
        cur_out = []
        cur_qual = []

        while True:
            item = qin.get()
            if item is None or item == "wait":
                break

            bounds, b_out, crop = item
            b_out = b_out.cpu().numpy()

            for bound, out, c in zip(bounds, b_out, crop):
                tc += 1
                print("bo", tc, bound, out.shape, c)
                if bound is not None:
                    if len(cur_out) > 0:
                        qout.put((cur_name, "".join(cur_out), "".join(cur_qual)))
                    cur_out = []
                    cur_qual = []
                    cur_name = bound
                    
                seq, qual = decoder.beam_search(np.ascontiguousarray(out), beam_size, beam_cut_threshold)
                # TODO: correct write
                cur_out.append(seq)
                cur_qual.append(qual)
        if len(cur_out) > 0:
            qout.put((cur_name, "".join(cur_out), "".join(cur_qual)))

def batch_process(qin, qout, b_len, s_len, pad):
    item = "wait"
    while item != None:
        data, crop, bounds = [], [], []
        while True:
            try:
                item = qin.get(timeout=1)
            except queue.Empty:
                item = "wait"
                break
            if item is None:
                break
            signal, metadata = item
            rescaled_signal = rescale_signal(signal).reshape(1, len(signal), 1)
            qout.put(([metadata], rescaled_signal, [slice(0, len(signal))]))
        qout.put("wait")

    qout.put(None)


 


class Basecaller:
    def __init__(self, model_file, input_q, output_q, pad=15, beam_size=5, beam_cut_threshold=0.1):
        b_len, s_len = 10, 3500

        self.input_q = input_q
        self.call_q = mp.Queue(100)
        self.final_q = mp.Queue()
        self.output_q = output_q

        self.batcher_proc = mp.Process(target=batch_process, args=(self.input_q, self.call_q, b_len, s_len, pad)) 
        self.caller_proc = mp.Process(target=caller_process, args=(model_file, self.call_q, self.final_q))
        self.final_proc = mp.Process(target=finalizer_process, args=(model_file, self.final_q, self.output_q, beam_size, beam_cut_threshold)) 

        self.batcher_proc.start()
        self.caller_proc.start()
        self.final_proc.start()


    def terminate(self):
        print("terminating")
        self.final_proc.join(1)
        self.caller_proc.join(1)
        self.batcher_proc.join(1)
        # use force if necessary (usually after Ctrl-C)
        self.final_proc.terminate()
        self.caller_proc.terminate()
        self.batcher_proc.terminate()
        self.input_q.cancel_join_thread()
        self.output_q.cancel_join_thread()

