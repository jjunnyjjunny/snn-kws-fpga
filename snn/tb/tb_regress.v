// tb_regress.v — regression: N clips (feat.hex [N,T,40]) each with clip_start, compares final class and
// output sums with pred.hex / vsum.hex. Prints "REGRESS n=.. match=.. sum_match=.. PASS/FAIL".
`timescale 1ns/1ps
`include "tb_cfg.vh"
module tb_regress;
    localparam N_IN = 40, NC = `N_CLASS, T = `T_FRAMES, N = `N_SAMPLES;
    reg clk = 0, rst = 1, clip_start = 0, feat_valid = 0, feat_last = 0;
    reg [7:0] feat_data = 0;
    wire frame_done, busy, det_valid; wire [3:0] class_idx; wire [NC*32-1:0] vo_sum;
    snn_core #(.N_CLASS(NC), .ZERO_SKIP(`ZERO_SKIP), .MEM_PREFIX(`MEM_PREFIX)) dut (
        .clk(clk), .rst(rst), .clip_start(clip_start), .feat_valid(feat_valid), .feat_data(feat_data),
        .feat_last(feat_last), .frame_done(frame_done), .class_idx(class_idx), .vo_sum(vo_sum), .det_valid(det_valid), .busy(busy));
    always #5 clk = ~clk;
    reg [7:0]  feat_mem [0:N*T*N_IN-1];
    reg [3:0]  pred_mem [0:N-1];
    reg [31:0] vsum_mem [0:N*NC-1];
    integer n, t, p, c, match, sum_match, bad, tot_cyc;
    initial begin
        $readmemh({`VEC_DIR, "/feat.hex"}, feat_mem); $readmemh({`VEC_DIR, "/pred.hex"}, pred_mem); $readmemh({`VEC_DIR, "/vsum.hex"}, vsum_mem);
        match = 0; sum_match = 0; tot_cyc = 0;
        repeat (3) @(posedge clk); rst <= 0; @(posedge clk); while (busy) @(posedge clk);
        for (n = 0; n < N; n = n + 1) begin
            clip_start <= 1; @(posedge clk); clip_start <= 0; @(posedge clk); while (busy) @(posedge clk);
            for (t = 0; t < T; t = t + 1) begin
                for (p = 0; p < N_IN; p = p + 1) begin
                    feat_valid <= 1; feat_data <= feat_mem[(n*T + t)*N_IN + p]; feat_last <= (p == N_IN-1); @(posedge clk);
                end
                feat_valid <= 0; feat_last <= 0;
                while (!frame_done) @(posedge clk); @(posedge clk);
                tot_cyc = tot_cyc + dut.u_ctrl.cyc_frame;
            end
            bad = 0;
            for (c = 0; c < NC; c = c + 1) if ($signed(vo_sum[c*32 +: 32]) !== $signed(vsum_mem[n*NC + c])) bad = bad + 1;
            if (bad == 0) sum_match = sum_match + 1;
            if (class_idx == pred_mem[n]) match = match + 1;
            else $display("MISMATCH sample=%0d got=%0d exp=%0d", n, class_idx, pred_mem[n]);
            if (`VERBOSE) $display("sample %0d: class=%0d exp=%0d sum_bad=%0d", n, class_idx, pred_mem[n], bad);
        end
        $display("REGRESS n=%0d match=%0d sum_match=%0d cyc_avg=%0d %s", N, match, sum_match, tot_cyc / (N*T), (match == N && sum_match == N) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
