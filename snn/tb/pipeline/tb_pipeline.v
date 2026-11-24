// tb_pipeline.v — integration test: pcm24 stream -> kws_top (fe_top + vad + preroll + snn_core + decision_seg)
// compared frame by frame with pipeline_int_ref.py (expected.hex: active/seg_start/seg_end/det/fed, seg_sums.hex).
// Samples are spaced `SAMPLE_CYC cycles apart; feeding stalls while the SNN replays (keeps frame events ordered,
// equivalent to real time where a frame period (480k cycles) always exceeds the replay time).
`timescale 1ns/1ps
`include "tb_cfg.vh"
`ifndef GATE_OFF
`define GATE_OFF 1'b0   // 1 = clock gating disabled (A/B check: results must be identical)
`endif
module tb_pipeline;
    localparam NC = 12;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, new_sample = 0; reg signed [23:0] pcm = 0;
    wire fdone, vad_active, det_valid, det_any, snn_active, overrun; wire [15:0] energy; wire [3:0] det_class; wire [NC*32-1:0] seg_sum;
    wire [31:0] cnt_frames, cnt_snn_frames, cnt_busy, cnt_total;
    kws_top #(.N_CLASS(NC), .ENERGY_SEL(`ENERGY_SEL), .T_ON(`T_ON), .T_OFF(`T_OFF), .N_ON(`N_ON), .H(`H), .P(`P), .L_MIN(`L_MIN), .L_MAX(`L_MAX),
              .ROM_DIR(`ROM_DIR), .MEM_PREFIX(`MEM_PREFIX)) dut (
        .clk(clk), .rst(rst), .new_sample(new_sample), .pcm(pcm), .margin_sel(2'd`MARGIN_SEL), .gate_off(`GATE_OFF),
        .ext_en(1'b0), .ext_clip_start(1'b0), .ext_valid(1'b0), .ext_data(8'd0), .ext_last(1'b0), .snn_frame_done(), .snn_class(),
        .fe_frame_done(fdone), .energy(energy), .vad_active(vad_active),
        .det_valid(det_valid), .det_any(det_any), .det_class(det_class), .snn_active(snn_active),
        .cnt_frames(cnt_frames), .cnt_snn_frames(cnt_snn_frames), .cnt_snn_busy_cycles(cnt_busy), .cnt_total_cycles(cnt_total), .seg_sum(seg_sum), .overrun(overrun));
    reg [23:0] pcm_mem [0:`N_SAMPLES-1]; reg [15:0] exp_mem [0:`T_FRAMES-1]; reg [15:0] e_mem [0:`T_FRAMES-1]; reg [31:0] sums_mem [0:`N_DETS*NC-1];
    integer i, t, mism, fed, ndet, c, sum_mism, max_replay, replay_cyc;
    reg [15:0] got; reg seen_start, seen_end, seen_det_any, seen_det_valid; reg [3:0] got_class;
    // per-frame bookkeeping: events between two fe_frame_done pulses belong to the frame just finished
    always @(posedge clk) begin
        if (dut.seg_start) seen_start <= 1;
        if (dut.seg_end) seen_end <= 1;
        if (det_any) begin seen_det_any <= 1; got_class <= det_class; seen_det_valid <= det_valid;
            for (c = 0; c < NC; c = c + 1) if ($signed(seg_sum[c*32 +: 32]) !== $signed(sums_mem[ndet*NC + c])) sum_mism = sum_mism + 1;
            ndet = ndet + 1; end
        if (dut.pb_frame_done) fed = fed + 1;
        if (snn_active) replay_cyc = replay_cyc + 1;
    end
    task check_frame; begin
        got = {7'd0, got_class, seen_det_valid, seen_det_any, seen_end, seen_start, vad_active};   // bit layout = expected.hex
        if (got[8:0] !== exp_mem[t][8:0] || fed !== (exp_mem[t] >> 9) || energy !== e_mem[t]) begin
            mism = mism + 1;
            if (mism <= 8) $display("MISMATCH frame %0d: exp active=%0d start=%0d end=%0d det_any=%0d det_valid=%0d class=%0d fed=%0d E=%0d | got active=%0d start=%0d end=%0d det_any=%0d det_valid=%0d class=%0d fed=%0d E=%0d",
                t, exp_mem[t][0], exp_mem[t][1], exp_mem[t][2], exp_mem[t][3], exp_mem[t][4], exp_mem[t][8:5], exp_mem[t] >> 9, e_mem[t], vad_active, seen_start, seen_end, seen_det_any, seen_det_valid, got_class, fed, energy);
        end
        if (replay_cyc > max_replay) max_replay = replay_cyc;
        seen_start = 0; seen_end = 0; seen_det_any = 0; seen_det_valid = 0; got_class = 0; fed = 0; replay_cyc = 0; t = t + 1;
    end endtask
    initial begin
        $readmemh({`VEC_DIR, "/pcm24.hex"}, pcm_mem); $readmemh({`VEC_DIR, "/expected.hex"}, exp_mem); $readmemh({`VEC_DIR, "/energy.hex"}, e_mem);
        if (`N_DETS > 0) $readmemh({`VEC_DIR, "/seg_sums.hex"}, sums_mem);
        t = 0; mism = 0; fed = 0; ndet = 0; sum_mism = 0; max_replay = 0; replay_cyc = 0; seen_start = 0; seen_end = 0; seen_det_any = 0; seen_det_valid = 0; got_class = 0;
        repeat (3) @(posedge clk); rst <= 0; repeat (3) @(posedge clk);
        for (i = 0; i < `N_SAMPLES; i = i + 1) begin
            while (dut.st != 0 || dut.pb_busy || dut.snn_busy || dut.dec_busy) @(posedge clk);   // hold samples only while a replay/decision is in progress (not during the whole VAD segment)
            pcm <= $signed(pcm_mem[i]); new_sample <= 1; @(posedge clk); new_sample <= 0;
            repeat (`SAMPLE_CYC - 1) begin @(posedge clk);
                if (fdone) begin   // let the VAD/sequencer react, wait for the frame's replay + decision, then compare
                    repeat (6) @(posedge clk); while (dut.st != 0 || dut.pb_busy || dut.snn_busy || dut.dec_busy) @(posedge clk); repeat (4) @(posedge clk); check_frame;
                end
            end
        end
        // last frame: wait for its frame_done (and any replay/decision), then check it
        begin : last_frame
            integer w; w = 0;
            while (!fdone && w < 40000) begin @(posedge clk); w = w + 1; end
            if (fdone) begin repeat (4) @(posedge clk); while (dut.st != 0 || dut.pb_busy || dut.snn_busy || dut.dec_busy) @(posedge clk); repeat (4) @(posedge clk); check_frame; end
        end
        repeat (100) @(posedge clk);
        $display("PIPELINE vec=%s frames=%0d/%0d mism=%0d dets=%0d/%0d sum_mism=%0d overrun=%0d snn_frames=%0d duty=%0d%% max_replay_cyc=%0d %s", `VEC_DIR, t, `T_FRAMES, mism, ndet, `N_DETS, sum_mism, overrun,
                 cnt_snn_frames, (100 * cnt_snn_frames) / (t > 0 ? t : 1), max_replay, (t == `T_FRAMES && mism == 0 && ndet == `N_DETS && sum_mism == 0 && !overrun) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
