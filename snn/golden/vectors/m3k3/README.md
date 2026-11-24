# Test vectors for `m3k3` (M3)

One directory per clip. Each `.hex` file: 2 comment lines then one hex word per line,
row-major flattening of the shape given in the header (row = frame/timestep). Signed values are two's complement.

| file | content |
|---|---|
| `pcm24.hex` | 16000 x 24-bit input samples |
| `x_db.hex` | after >>6 scaling (18 bit) |
| `windowed.hex` | [T,WIN] windowed frames |
| `fft_re.hex / fft_im.hex` | [T,512] FFT output |
| `power.hex` | [T,257] power |
| `mel_acc.hex` | [T,n_mels] mel accumulator |
| `log2q.hex` | [T,n_mels] log2 Q6.4 |
| `feat.hex` | [T,n_mels] 8-bit feature (SNN input) |
| `<layer>_v.hex / _s.hex` | [T,N] membrane (after reset) / spikes |
| `out_v.hex` | [T,C] output membrane |
| `info.json` | label, predictions (sum / last / max), output sums |
