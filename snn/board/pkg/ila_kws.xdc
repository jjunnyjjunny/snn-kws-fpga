# ila_kws.xdc — (documentation only) the KWS ILA is an instantiated IP (ila_kws, created in build_bitstream.tcl) inside
# user_logic_inst; nothing to constrain here. Probe map: see the comment block at the end of pkg/user_logic.v.
# Hardware Manager: for mode 3 audio capture set capture mode "qualified" with probe1 (dbg_cap_valid) == 1, depth 16384
# (= 1.024 s at 16 kHz) and export probe2 (dbg_cap_sample[17:0]) as CSV; use "all" for the other probes.
