"""
gen_codec_variants.py — generate reg_set_init_16k_v*.c (ADAU1761 input-gain experiments, B-2) from reg_set_init_16k.c.
Each variant only changes register VALUES in the reference write sequence and/or inserts extra writes right after the
'Record mic bias' write (so every write still happens before the clock-enable registers), and appends read-backs of the
touched registers at the end. No bitstream change. Encodings (ADAU1761 data sheet register map; cross-checked with the
Linux adau1761.c driver shifts/TLVs):
  0x400A/0x400C Record Mixer L/R Control 0: [6:4] LINPG/RINPG, [3:1] LINNG/RINNG (000 mute, 001 -12 dB, 010 -9, 011 -6,
                100 -3, 101 0 dB, 110 +3, 111 +6 dB), [0] MX1EN/MX2EN
  0x400B/0x400D Record Mixer L/R Control 1: [4:3] LDBOOST/RDBOOST (00 mute, 01 0 dB, 10 +20 dB), [2:0] MX1AUXG/MX2AUXG
                (same 3-bit gain code as above, LAUX/RAUX single-ended line input)
  0x400E/0x400F L/R Differential Input Volume: [7:2] LDVOL/RDVOL (-12 dB + 0.75 dB x code; 16 = 0 dB, 32 = +12 dB,
                63 = +35.25 dB), [1] LDMUTE (1 = unmuted), [0] LDEN (1 = PGA enabled)
  0x4010 Record Microphone Bias: [3] MPERF, [2] MBI (0 = 0.90 AVDD, 1 = 0.65 AVDD), [0] MBIEN
  0x4019 ADC Control: [5] HPF (1 = ADC high-pass on), [4] ADCPOL? / [3] DMPOL, [2] INSEL/DMSW, [1:0] ADCEN (11 = both)
  0x4011..0x4014 ALC (vD only): 0x4011 [5:3] ALCMAX, [2:0] ALCSEL (000 off, 011 stereo); 0x4012 [7:4] ALCHOLD,
                [3:0] ALCTARG; 0x4013 [7:4] ALCATCK, [3:0] ALCDEC; 0x4014 [5] NGEN, [4:0] NGTHR  (bit meanings per the
                data sheet 'ALC' section — verify before relying on vD).
usage: python gen_codec_variants.py   (writes reg_set_init_16k_vA.c ... in this directory)
"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'reg_set_init_16k.c')).read()

def write_block(addr, val, comment):
    return (f"    // {comment}\n    writeData[0] = 0x40; writeData[1] = 0x{addr & 0xff:02X};\n    writeData[2] = 0x{val:02X};\n"
            f"    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);\n    usleep(50000);\n\n")

def read_block(addr, expect, comment):
    return (f"    // {comment}\n    writeData[0] = 0x40; writeData[1] = 0x{addr & 0xff:02X};\n"
            f"    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);\n    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);\n"
            f"    xil_printf(\"Address: %02x%02x, Read Data: %02x  ({comment}, expect {expect:02X})\\r\\n\", *writeData, *(writeData+1), *readData);\n    usleep(50000);\n\n")

def make(name, title, overrides, inserts, notes):
    """overrides: {comment_in_reference: new_value}; inserts: [(addr, val, comment)] after the mic-bias write."""
    s = SRC.replace(" * reg_set_init_16k.c: reference ADAU1761 init (48 kHz) + sample-rate change to 16 kHz for the KWS design",
                    f" * reg_set_init_16k_{name}.c: {title}\n *   " + "\n *   ".join(notes))
    # PLL-lock poll before the core clock is enabled (the 48 kHz cold run read 0x400B/0x400D back as 00 although 05 was
    # written: analog/record registers written before the clock is stable can be lost; the warm 16 kHz run read 05)
    pll = ("    // KWS: wait for PLL lock (0x4002 byte 6 bit 0) before enabling the core clock and writing the record/playback registers\n"
           "    { int i; for (i = 0; i < 100; i++) { writeData[0] = 0x40; writeData[1] = 0x02;\n"
           "        XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B); XIicPs_MasterRecvPolled(&IicPsInstance, readData, 6, 0x3B);\n"
           "        if (readData[5] & 0x01) break; usleep(10000); }\n"
           "      xil_printf(\"PLL lock poll: %d x 10 ms, 0x4002 byte 6 = %02x (bit0 = lock)\\r\\n\", i, readData[5]); }\n\n")
    core = "    // Clock Control\n    writeData[0] = 0x40; writeData[1] = 0x00;\n    writeData[2] = 0x0F;"
    assert core in s; s = s.replace(core, pll + core, 1)
    touched = []
    for cm, val in overrides.items():
        pat = re.compile(r"(    // " + re.escape(cm) + r"\n    writeData\[0\] = 0x40; writeData\[1\] = 0x([0-9A-Fa-f]{2});\n    writeData\[2\] = )0x[0-9A-Fa-f]{2};")
        m = pat.search(s); assert m, cm
        s = s[:m.start()] + m.group(1) + f"0x{val:02X};" + s[m.end():]
        touched.append((int(m.group(2), 16) | 0x4000, val, cm))
    ins = ''.join(write_block(a, v, c) for a, v, c in inserts)
    anchor = "    // Play Mixer Left 0\n    writeData[0] = 0x40; writeData[1] = 0x1C;"
    assert anchor in s; s = s.replace(anchor, ins + anchor, 1)
    touched += [(a, v, c) for a, v, c in inserts]
    rb = ''.join(read_block(a, v, c) for a, v, c in touched)
    tail = "    xil_printf(\"KWS: codec configured for 16 kHz"
    assert tail in s
    s = s.replace(tail, rb + f"    xil_printf(\"KWS variant {name}: {title}\\r\\n\");\n" + tail, 1)
    open(os.path.join(HERE, f'reg_set_init_16k_{name}.c'), 'w').write(s)
    print(f"reg_set_init_16k_{name}.c  touched: " + ', '.join(f"0x{a:04X}={v:02X}" for a, v, _ in touched))

# v0: control = reference 16 kHz sequence + PLL-lock poll only (no gain change); read-backs of the record registers
make('v0', 'control: reference 16 kHz sequence + PLL-lock poll, no gain change',
     {'Rec Mixer Left 0': 0x0B, 'Rec Mixer Right 0': 0x0B, 'Rec Mixer Left 1': 0x05, 'Rec Mixer Right 1': 0x05, 'ADC control': 0x03}, [],
     ['identical register values to reg_set_init_16k.c; only the PLL-lock wait is added before Clock Control = 0x0F',
      'use it to check whether 0x400B/0x400D read back 05 on a cold power-up'])
# vA: keep the reference path (LINN/RINN single-ended into the record mixer, PGA muted), mixer gain 0 -> +6 dB (max),
#     line-in AUX muted (removes the open LINE IN jack noise), ADC HPF on (DC offset ~ -1200 LSB18 in the captures)
make('vA', 'reference input path, record-mixer LINN/RINN gain +6 dB (max), AUX muted, ADC HPF on',
     {'Rec Mixer Left 0': 0x0F, 'Rec Mixer Right 0': 0x0F, 'Rec Mixer Left 1': 0x00, 'Rec Mixer Right 1': 0x00, 'ADC control': 0x23},
     [], ['0x400A/0x400C = 0x0F: LINNG/RINNG = 111 (+6 dB), MX1EN; 0x400B/0x400D = 0x00: PGA boost mute, AUX mute',
          '0x4019 = 0x23: ADC HPF on (bit 5) + both ADCs; expected level +6 dB vs reg_set_init_16k.c'])
# vB: differential PGA path (LINP-LINN / RINP-RINN) at 0 dB boost + LDVOL +12 dB; mixer LINN input muted so the mic
#     reaches the ADC only through the PGA (if the jack drives LINN with LINP ac-grounded, the PGA sees the mic single-ended)
make('vB', 'PGA path: boost 0 dB + differential volume +12 dB, mixer LINN/AUX muted, ADC HPF on',
     {'Rec Mixer Left 0': 0x01, 'Rec Mixer Right 0': 0x01, 'Rec Mixer Left 1': 0x08, 'Rec Mixer Right 1': 0x08, 'ADC control': 0x23},
     [(0x400E, (32 << 2) | 0x03, 'Left differential input volume: LDVOL +12 dB, unmute, enable'),
      (0x400F, (32 << 2) | 0x03, 'Right differential input volume: RDVOL +12 dB, unmute, enable')],
     ['0x400A/0x400C = 0x01: mixer inputs muted (MX1EN only); 0x400B/0x400D = 0x08: LDBOOST = 01 (0 dB), AUX mute',
      '0x400E/0x400F = 0x83: LDVOL code 32 = +12 dB, LDMUTE = 1 (unmuted), LDEN = 1; expected level about +12 dB'])
# vC: PGA path with +20 dB boost + LDVOL +6 dB (about +26 dB), for the 20-30 cm case (-35 dBFS peak -> about -9 dBFS)
make('vC', 'PGA path: boost +20 dB + differential volume +6 dB (about +26 dB), mixer LINN/AUX muted, ADC HPF on',
     {'Rec Mixer Left 0': 0x01, 'Rec Mixer Right 0': 0x01, 'Rec Mixer Left 1': 0x10, 'Rec Mixer Right 1': 0x10, 'ADC control': 0x23},
     [(0x400E, (24 << 2) | 0x03, 'Left differential input volume: LDVOL +6 dB, unmute, enable'),
      (0x400F, (24 << 2) | 0x03, 'Right differential input volume: RDVOL +6 dB, unmute, enable')],
     ['0x400B/0x400D = 0x10: LDBOOST = 10 (+20 dB), AUX mute; 0x400E/0x400F = 0x63: LDVOL code 24 = +6 dB, unmuted, enabled',
      'expected level about +26 dB vs reg_set_init_16k.c (near 3 cm may clip: watch peak and listen for distortion)'])
# vD: vC path + ALC (hardware AGC on the PGA): stereo, max gain +30 dB-ish, target about -6 dBFS, medium attack/decay,
#     noise gate on so silence is not pumped up. Bit encodings of 0x4011-0x4014 must be checked against the data sheet.
make('vD', 'PGA path + ALC (codec AGC): boost +20 dB, ALC stereo, noise gate on (bits of 0x4011-0x4014 to verify)',
     {'Rec Mixer Left 0': 0x01, 'Rec Mixer Right 0': 0x01, 'Rec Mixer Left 1': 0x10, 'Rec Mixer Right 1': 0x10, 'ADC control': 0x23},
     [(0x400E, (16 << 2) | 0x03, 'Left differential input volume: LDVOL 0 dB (ALC takes over), unmute, enable'),
      (0x400F, (16 << 2) | 0x03, 'Right differential input volume: RDVOL 0 dB, unmute, enable'),
      (0x4011, (0b110 << 3) | 0b011, 'ALC Control 0: ALCMAX code 110, ALCSEL 011 = stereo'),
      (0x4012, (0b0010 << 4) | 0b1100, 'ALC Control 1: ALCHOLD code 0010, ALCTARG code 1100 (about -6 dB)'),
      (0x4013, (0b0011 << 4) | 0b0100, 'ALC Control 2: ALCATCK code 0011, ALCDEC code 0100'),
      (0x4014, (1 << 5) | 0b01000, 'ALC Control 3: NGEN = 1, NGTHR code 01000')],
     ['same analog path as vC with LDVOL 0 dB; ALC enabled on both channels (0x4011 ALCSEL = 011) with noise gate',
      'ALC field encodings in 0x4011-0x4014 are from memory of the data sheet ALC section: VERIFY before use'])
