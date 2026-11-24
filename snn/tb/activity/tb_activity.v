// tb_activity.v — activity counting on a scenario stream (kws_top with -DACTIVITY): cycles per block, BRAM accesses,
// DSP multiplies, SNN/front-end duty. Also checks the frame-level golden (expected.hex) like tb_pipeline.
`timescale 1ns/1ps
`include "tb_cfg.vh"
`ifndef GATE_OFF
`define GATE_OFF 1'b0   // 1 = clock gating disabled (A/B check: results must be identical)
`endif
module tb_activity;
    localparam NC = 12;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1, new_sample = 0; reg signed [23:0] pcm = 0;
    wire fdone, vad_active, det_valid, det_any, snn_active, overrun, fe_gated; wire [15:0] energy; wire [3:0] det_class; wire [NC*32-1:0] seg_sum;
    wire [31:0] cnt_frames, cnt_snn_frames, cnt_busy, cnt_total, cnt_fe_frames, cnt_fe_gated, cnt_fe_ce, cnt_snn_ce;
    kws_top #(.N_CLASS(NC), .ENERGY_SEL(1), .T_ON(`T_ON), .T_OFF(`T_OFF), .N_ON(`N_ON), .H(`H), .P(`P), .L_MIN(`L_MIN), .L_MAX(`L_MAX), .T0(`T0),
              .ROM_DIR(`ROM_DIR), .MEM_PREFIX(`MEM_PREFIX)) dut (
        .clk(clk), .rst(rst), .new_sample(new_sample), .pcm(pcm), .margin_sel(2'd0), .gate_off(`GATE_OFF),
        .ext_en(1'b0), .ext_clip_start(1'b0), .ext_valid(1'b0), .ext_data(8'd0), .ext_last(1'b0), .snn_frame_done(), .snn_class(),
        .fe_frame_done(fdone), .energy(energy), .vad_active(vad_active), .det_valid(det_valid), .det_any(det_any), .det_class(det_class), .snn_active(snn_active),
        .cnt_frames(cnt_frames), .cnt_snn_frames(cnt_snn_frames), .cnt_snn_busy_cycles(cnt_busy), .cnt_total_cycles(cnt_total), .seg_sum(seg_sum), .overrun(overrun),
        .fe_gated(fe_gated), .cnt_fe_frames(cnt_fe_frames), .cnt_fe_gated(cnt_fe_gated), .cnt_fe_ce_cycles(cnt_fe_ce), .cnt_snn_ce_cycles(cnt_snn_ce));
    reg [23:0] pcm_mem [0:`N_SAMPLES-1]; reg [15:0] exp_mem [0:`T_FRAMES-1];
    integer i, t, mism, ndet, nhit, dets_ok;
    reg seen_det_any, seen_det_valid; reg [3:0] got_class;
    always @(posedge clk) if (det_any) begin seen_det_any <= 1; got_class <= det_class; seen_det_valid <= det_valid; ndet = ndet + 1; end
    task check_frame; begin
        if ({seen_det_valid, seen_det_any} !== {exp_mem[t][4], exp_mem[t][3]} || (seen_det_any && got_class !== exp_mem[t][8:5])) mism = mism + 1;
        seen_det_any = 0; seen_det_valid = 0; got_class = 0; t = t + 1;
    end endtask
    initial begin
        $readmemh({`VEC_DIR, "/pcm24.hex"}, pcm_mem); $readmemh({`VEC_DIR, "/expected.hex"}, exp_mem);
        t = 0; mism = 0; ndet = 0; seen_det_any = 0; seen_det_valid = 0; got_class = 0;
        repeat (3) @(posedge clk); rst <= 0; repeat (3) @(posedge clk);
        for (i = 0; i < `N_SAMPLES; i = i + 1) begin
            while (dut.st != 0 || dut.pb_busy || dut.snn_busy || dut.dec_busy) @(posedge clk);
            pcm <= $signed(pcm_mem[i]); new_sample <= 1; @(posedge clk); new_sample <= 0;
            repeat (`SAMPLE_CYC - 1) begin @(posedge clk);
                if (fdone) begin repeat (6) @(posedge clk); while (dut.st != 0 || dut.pb_busy || dut.snn_busy || dut.dec_busy) @(posedge clk); repeat (4) @(posedge clk); check_frame; end
            end
        end
        repeat (40000) @(posedge clk);
        // ---- activity report (cycle-stalled feeding: use per-frame normalisation = per second of audio)
        $display("ACTIVITY vec=%s T0=%0d vad=%0d frames=%0d gated=%0d snn_frames=%0d mism=%0d dets=%0d overrun=%0d", `VEC_DIR, `T0, `VAD_ON, cnt_fe_frames, cnt_fe_gated, cnt_snn_frames, mism, ndet, overrun);
`ifdef ACTIVITY
        $display("ACT fe_cyc=%0d win_cyc=%0d fft_cyc=%0d mel_cyc=%0d fe_buf_rd=%0d fe_buf_wr=%0d ring_wr=%0d win_mult=%0d fft_mult=%0d mel_mult=%0d",
                 dut.act_fe_cyc, dut.u_fe.u_win.act_cyc, dut.u_fe.u_fft.act_cyc, dut.u_fe.u_mel.act_cyc, dut.u_fe.u_buf.act_rd, dut.u_fe.u_buf.act_wr, dut.u_fe.u_ring.act_wr,
                 dut.u_fe.u_win.act_mult, dut.u_fe.u_fft.act_mult, dut.u_fe.u_mel.act_mult);
        $display("ACT snn_cyc=%0d snn_rom_rd=%0d snn_mult=%0d snn_vmem_wr=%0d pb_cyc=%0d dec_cyc=%0d total_cyc=%0d",
                 dut.u_snn.act_cyc, dut.u_snn.act_rom_rd, dut.u_snn.act_mult, dut.u_snn.act_vmem_wr, dut.act_pb_cyc, dut.act_dec_cyc, dut.act_total);
        $display("ACT fe_ce_cyc=%0d snn_ce_cyc=%0d total_cyc=%0d", cnt_fe_ce, cnt_snn_ce, cnt_total);
`endif
        $finish;
    end
endmodule
