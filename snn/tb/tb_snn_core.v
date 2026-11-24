// tb_snn_core.v — golden-vector test bench for snn_core.
// Configuration comes from tb_cfg.vh (written by run_vec.sh): `VEC_DIR, `EXP_PRED, `STOP_ON_MISMATCH,
// `T_FRAMES, `N_CLASS, `ZERO_SKIP, `MEM_PREFIX, `CLIP_START (1 = pulse clip_start before frame 0).
// Per frame it compares every hidden membrane (after reset), every spike, every output membrane
// with <layer>_v.hex / <layer>_s.hex / out_v.hex, then the final class with EXP_PRED.
`timescale 1ns/1ps
`include "tb_cfg.vh"
module tb_snn_core;
    localparam N_IN = 40, N_HID = 64, NC = `N_CLASS, T = `T_FRAMES;
    reg clk = 0, rst = 1, clip_start = 0, feat_valid = 0, feat_last = 0;
    reg [7:0] feat_data = 0;
    wire frame_done, busy, det_valid;
    wire [3:0] class_idx;
    wire [NC*32-1:0] vo_sum;
    snn_core #(.N_CLASS(NC), .ZERO_SKIP(`ZERO_SKIP), .MEM_PREFIX(`MEM_PREFIX)) dut (
        .clk(clk), .rst(rst), .clip_start(clip_start), .feat_valid(feat_valid), .feat_data(feat_data),
        .feat_last(feat_last), .frame_done(frame_done), .class_idx(class_idx), .vo_sum(vo_sum),
        .det_valid(det_valid), .busy(busy));
    always #5 clk = ~clk;
    // expected data
    reg [7:0]  feat_mem [0:T*N_IN-1];
    reg [0:0]  s1 [0:T*N_HID-1]; reg [0:0] s2 [0:T*N_HID-1]; reg [0:0] s3 [0:T*N_HID-1]; reg [0:0] s4 [0:T*N_HID-1];
    reg [15:0] v1 [0:T*N_HID-1]; reg [15:0] v2 [0:T*N_HID-1]; reg [15:0] v3 [0:T*N_HID-1]; reg [15:0] v4 [0:T*N_HID-1];
    reg [23:0] ov [0:T*NC-1];
    integer t, p, c, mism, frame_mism, tot_cyc, tot_l1, tot_hid, tot_out, tot_upd, tot_dec, tot_skip, tot_run1, tot_runh, tot_runo;
    integer max_cyc;
    reg [15:0] exp_v; reg exp_s; reg [23:0] exp_o;
    task check(input [8*8-1:0] what, input integer layer, input integer idx, input integer expv, input integer gotv);
        begin
            if (expv !== gotv) begin
                mism = mism + 1; frame_mism = frame_mism + 1;
                if (`STOP_ON_MISMATCH || mism <= 5)
                    $display("MISMATCH vec=%s frame=%0d layer=%0d %0s idx=%0d expected=%0d got=%0d", `VEC_DIR, t, layer, what, idx, expv, gotv);
                if (`STOP_ON_MISMATCH) begin $display("RESULT vec=%s STOP mism=%0d", `VEC_DIR, mism); $finish; end
            end
        end
    endtask
    initial begin
        $readmemh({`VEC_DIR, "/feat.hex"}, feat_mem);
        $readmemh({`VEC_DIR, "/tcn1_s.hex"}, s1); $readmemh({`VEC_DIR, "/tcn2_s.hex"}, s2);
        $readmemh({`VEC_DIR, "/tcn3_s.hex"}, s3); $readmemh({`VEC_DIR, "/tcn4_s.hex"}, s4);
        $readmemh({`VEC_DIR, "/tcn1_v.hex"}, v1); $readmemh({`VEC_DIR, "/tcn2_v.hex"}, v2);
        $readmemh({`VEC_DIR, "/tcn3_v.hex"}, v3); $readmemh({`VEC_DIR, "/tcn4_v.hex"}, v4);
        $readmemh({`VEC_DIR, "/out_v.hex"}, ov);
        mism = 0; tot_cyc = 0; tot_l1 = 0; tot_hid = 0; tot_out = 0; tot_upd = 0; tot_dec = 0; tot_skip = 0; tot_run1 = 0; tot_runh = 0; tot_runo = 0; max_cyc = 0;
        repeat (3) @(posedge clk); rst <= 0;
        @(posedge clk); while (busy) @(posedge clk);
        if (`CLIP_START) begin clip_start <= 1; @(posedge clk); clip_start <= 0; @(posedge clk); while (busy) @(posedge clk); end
        for (t = 0; t < T; t = t + 1) begin
            frame_mism = 0;
            for (p = 0; p < N_IN; p = p + 1) begin
                feat_valid <= 1; feat_data <= feat_mem[t*N_IN + p]; feat_last <= (p == N_IN-1);
                @(posedge clk);
            end
            feat_valid <= 0; feat_last <= 0;
            while (!frame_done) @(posedge clk);
            @(posedge clk);
            // ---- compare state after frame t
            for (p = 0; p < N_HID; p = p + 1) begin
                check("v", 1, p, $signed(v1[t*N_HID+p]), $signed(dut.v_mem[0*N_HID+p]));
                check("v", 2, p, $signed(v2[t*N_HID+p]), $signed(dut.v_mem[1*N_HID+p]));
                check("v", 3, p, $signed(v3[t*N_HID+p]), $signed(dut.v_mem[2*N_HID+p]));
                check("v", 4, p, $signed(v4[t*N_HID+p]), $signed(dut.v_mem[3*N_HID+p]));
                check("s", 1, p, s1[t*N_HID+p], dut.u_sh1.cur[p]);
                check("s", 2, p, s2[t*N_HID+p], dut.u_sh2.cur[p]);
                check("s", 3, p, s3[t*N_HID+p], dut.u_sh3.cur[p]);
                check("s", 4, p, s4[t*N_HID+p], dut.u_sh4.cur[p]);
            end
            for (c = 0; c < NC; c = c + 1)
                check("vo", 5, c, $signed(ov[t*NC+c]), $signed(dut.vo[c]));
            tot_cyc = tot_cyc + dut.u_ctrl.cyc_frame; if (dut.u_ctrl.cyc_frame > max_cyc) max_cyc = dut.u_ctrl.cyc_frame;
            tot_l1 = tot_l1 + dut.u_ctrl.cyc_acc_l1; tot_hid = tot_hid + dut.u_ctrl.cyc_acc_hid; tot_out = tot_out + dut.u_ctrl.cyc_acc_out;
            tot_upd = tot_upd + dut.u_ctrl.cyc_upd; tot_dec = tot_dec + dut.u_ctrl.cyc_dec;
            tot_skip = tot_skip + dut.u_ctrl.rows_skip_l1; tot_run1 = tot_run1 + dut.u_ctrl.rows_run_l1;
            tot_runh = tot_runh + dut.u_ctrl.rows_run_hid; tot_runo = tot_runo + dut.u_ctrl.rows_run_out;
            if (`VERBOSE) $display("frame %0d: cyc=%0d mism=%0d class=%0d", t, dut.u_ctrl.cyc_frame, frame_mism, class_idx);
        end
        $display("RESULT vec=%s mism=%0d pred=%0d exp=%0d %s cyc_total=%0d cyc_avg=%0d cyc_max=%0d l1=%0d hid=%0d out=%0d upd=%0d dec=%0d skip_l1=%0d run_l1=%0d run_hid=%0d run_out=%0d",
            `VEC_DIR, mism, class_idx, `EXP_PRED, (class_idx == `EXP_PRED && mism == 0) ? "PASS" : "FAIL",
            tot_cyc, tot_cyc / T, max_cyc, tot_l1 / T, tot_hid / T, tot_out / T, tot_upd / T, tot_dec / T, tot_skip / T, tot_run1 / T, tot_runh / T, tot_runo / T);
        for (c = 0; c < NC; c = c + 1) $display("  vo_sum[%0d] = %0d", c, $signed(vo_sum[c*32 +: 32]));
        $finish;
    end
endmodule
