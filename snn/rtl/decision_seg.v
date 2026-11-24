// decision_seg.v — segment decision: sum of the output membranes over the frames the SNN processed in the
// current VAD segment; at segment end: argmax (lowest index on ties), keyword filter (class < N_KEYWORD),
// optional top1-top2 margin. det_valid/det_class pulse one frame after seg_end processing.
`timescale 1ns/1ps
module decision_seg #(parameter N_CLASS = 12, CW = 4, N_KEYWORD = 10, MARGIN = 0, MARGIN1 = 2838, MARGIN2 = 6212, MARGIN3 = 9633) (
    input  wire clk, input wire rst,
    input  wire [1:0] margin_sel,                 // operating point: 0 normal (MARGIN) / 1 strict (MARGIN1) / 2 (MARGIN2) / 3 very strict (MARGIN3)
    input  wire seg_begin,                        // pulse: clear the sums (SNN reset)
    input  wire frame_acc,                        // pulse: add vo_cur (after the SNN finished a frame)
    input  wire [N_CLASS*24-1:0] vo_cur,
    input  wire finalize,                         // pulse: segment ended and all frames accumulated
    output reg  det_valid, output reg det_any,    // det_any = a decision was taken (even if silence/unknown)
    output reg  [CW-1:0] det_class,
    output wire [N_CLASS*32-1:0] seg_sum_flat,
    output reg  busy
);
    reg signed [31:0] sum [0:N_CLASS-1];
    reg [2:0] st; localparam S_IDLE=0, S_ACC=1, S_ARG=2, S_FIN=3;
    reg fin_pend;                                 // finalize arrived while a frame was still being accumulated
    reg [CW-1:0] c, best_idx; reg signed [31:0] best, second;
    wire signed [23:0] vo_c = vo_cur[c*24 +: 24];
    wire signed [32:0] margin = {best[31], best} - {second[31], second};
    wire signed [32:0] margin_thr = (margin_sel == 2'd1) ? MARGIN1 : (margin_sel == 2'd2) ? MARGIN2 : (margin_sel == 2'd3) ? MARGIN3 : MARGIN;
    genvar g; generate for (g = 0; g < N_CLASS; g = g + 1) begin : G assign seg_sum_flat[g*32 +: 32] = sum[g]; end endgenerate
    integer i;
    always @(posedge clk) begin
        det_valid <= 0; det_any <= 0;
        if (rst) begin st <= S_IDLE; busy <= 0; det_class <= 0; fin_pend <= 0; for (i = 0; i < N_CLASS; i = i + 1) sum[i] <= 0; end
        else case (st)
            S_IDLE: begin
                if (seg_begin) begin for (i = 0; i < N_CLASS; i = i + 1) sum[i] <= 0; end
                if (frame_acc) begin c <= 0; st <= S_ACC; busy <= 1; if (finalize) fin_pend <= 1; end
                else if (finalize || fin_pend) begin fin_pend <= 0; c <= 0; st <= S_ARG; busy <= 1; end
            end
            S_ACC: begin
                sum[c] <= sum[c] + {{8{vo_c[23]}}, vo_c};
                if (finalize) fin_pend <= 1;
                if (c == N_CLASS - 1) begin st <= S_IDLE; busy <= fin_pend | finalize; end else c <= c + 1'b1;
            end
            S_ARG: begin
                if (c == 0) begin best <= sum[0]; best_idx <= 0; second <= -32'sd2147483648; end
                else if (sum[c] > best) begin second <= best; best <= sum[c]; best_idx <= c; end
                else if (sum[c] > second) second <= sum[c];
                if (c == N_CLASS - 1) st <= S_FIN; else c <= c + 1'b1;
            end
            S_FIN: begin
                det_any <= 1; det_class <= best_idx;
                det_valid <= (best_idx < N_KEYWORD) && (margin >= margin_thr);
                st <= S_IDLE; busy <= 0;
            end
            default: st <= S_IDLE;
        endcase
    end
endmodule
