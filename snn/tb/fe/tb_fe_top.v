// tb_fe_top.v — fe_top test: pcm24.hex samples with new_sample pulses (spacing `SAMPLE_CYC cycles) ->
// 98 frames of features compared with feat.hex; also e_sum/e_log recorded and per-block cycles.
`timescale 1ns/1ps
`include "tb_cfg.vh"
`ifndef GATE_OFF
`define GATE_OFF 1'b0   // 1 = clock gating disabled (A/B check: results must be identical)
`endif
module tb_fe_top;
    localparam T = `T_FRAMES, NM = 40, NS = 16000;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, new_sample = 0; reg signed [23:0] pcm = 0;
    wire fv, fl, fdone, busy; wire [7:0] fd; wire [13:0] es; wire [9:0] el; wire [31:0] cf, cw, cff, cm;
    fe_top #(.MULT_SERIAL(`MULT_SERIAL), .ROM_DIR(`ROM_DIR), .T0(`T0)) dut (.clk(clk), .rst(rst), .new_sample(new_sample), .pcm(pcm), .gate_off(`GATE_OFF), .clk_ce(), .cnt_ce(),
        .feat_valid(fv), .feat_data(fd), .feat_last(fl), .e_sum(es), .e_log(el), .frame_done(fdone), .busy(busy), .gated(), .e0_out(), .cnt_frames(), .cnt_gated(),
        .cyc_frame(cf), .cyc_win(cw), .cyc_fft(cff), .cyc_mel(cm));
    reg [23:0] pcm_mem [0:NS-1]; reg [7:0] feat_e [0:T*NM-1];
    integer i, k, frames, fcnt, mism, tot_cyc, tot_w, tot_f, tot_m, max_cyc; reg [7:0] got [0:NM-1];
    integer esum_chk, esum_mism;
    always @(posedge clk) if (fv) begin got[fcnt] <= fd; fcnt <= fcnt + 1; end
    always @(posedge clk) if (fdone) begin
        esum_chk = 0; for (k = 0; k < NM; k = k + 1) begin
            if (got[k] !== feat_e[frames*NM + k]) begin mism = mism + 1; if (mism <= 5) $display("FEAT mismatch frame %0d band %0d exp %0d got %0d", frames, k, feat_e[frames*NM+k], got[k]); end
            esum_chk = esum_chk + feat_e[frames*NM + k];
        end
        if (es !== esum_chk[13:0]) esum_mism = esum_mism + 1;
        tot_cyc = tot_cyc + cf; tot_w = tot_w + cw; tot_f = tot_f + cff; tot_m = tot_m + cm; if (cf > max_cyc) max_cyc = cf;
        frames = frames + 1; fcnt = 0;
    end
    initial begin
        $readmemh({`VEC_DIR, "/pcm24.hex"}, pcm_mem); $readmemh({`VEC_DIR, "/feat.hex"}, feat_e);
        frames = 0; fcnt = 0; mism = 0; tot_cyc = 0; tot_w = 0; tot_f = 0; tot_m = 0; max_cyc = 0; esum_mism = 0;
        repeat (3) @(posedge clk); rst <= 0; repeat (3) @(posedge clk);
        for (i = 0; i < NS; i = i + 1) begin
            pcm <= $signed(pcm_mem[i]); new_sample <= 1; @(posedge clk); new_sample <= 0;
            repeat (`SAMPLE_CYC - 1) @(posedge clk);
        end
        repeat (20000) @(posedge clk);
        $display("FE_TOP vec=%s frames=%0d mism=%0d esum_mism=%0d %s cyc_avg=%0d cyc_max=%0d win=%0d fft=%0d mel=%0d sample_cyc=%0d", `VEC_DIR, frames, mism, esum_mism,
                 (frames == T && mism == 0 && esum_mism == 0) ? "PASS" : "FAIL", tot_cyc / (frames > 0 ? frames : 1), max_cyc, tot_w / (frames > 0 ? frames : 1), tot_f / (frames > 0 ? frames : 1), tot_m / (frames > 0 ? frames : 1), `SAMPLE_CYC);
        $finish;
    end
endmodule
