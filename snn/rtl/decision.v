// decision.v — windowed time-sum of the output membranes + argmax (lowest index on ties).
//   sum[c] += vo[c] - vo[c] of WIN_SUM frames ago (ring, only once WIN_SUM frames have been seen)
//   class_idx = argmax_c sum[c]  (strict '>' scan from c=0 -> numpy argmax tie rule)
// Streaming extensions (parameters only until B-4 fixes the rule): MARGIN (top1-top2), N_CONSEC
// (same class for N frames), REFRACT (frames suppressed after a detection), N_KEYWORD (classes
// >= N_KEYWORD, i.e. silence/unknown, never count as detections).
`timescale 1ns/1ps
module decision #(
    parameter N_CLASS   = 12,
    parameter CW        = 4,
    parameter WIN_SUM   = 98,
    parameter WIN_DEPTH = 128,
    parameter MARGIN    = 0,
    parameter N_CONSEC  = 1,
    parameter REFRACT   = 0,
    parameter N_KEYWORD = 10
) (
    input  wire                 clk,
    input  wire                 clr,
    input  wire                 start,          // one pulse per frame after the output update
    input  wire [N_CLASS*24-1:0] vo_flat,       // current output membranes
    output reg                  done,           // one-cycle pulse
    output reg  [CW-1:0]        class_idx,
    output wire [N_CLASS*32-1:0] sum_flat,
    output reg                  det_valid,
    output wire                 busy
);
    reg signed [31:0] sum [0:N_CLASS-1];
    reg        [23:0] ring [0:WIN_DEPTH*N_CLASS-1];   // BRAM: vo history for the window
    reg        [23:0] ring_q;
    reg [6:0]  wp;                                     // frame write pointer (mod WIN_DEPTH=128)
    reg [7:0]  win_cnt;                                // frames seen since clr (saturates at WIN_SUM)
    wire [6:0] old_p = wp - WIN_SUM[6:0];
    wire       sub_ok = (win_cnt >= WIN_SUM);
    reg [2:0]  st;  localparam S_IDLE=0, S_RD=1, S_UPD=2, S_ARG=3, S_FIN=4;
    reg [CW-1:0] c;
    reg signed [31:0] best, second;
    reg [CW-1:0] best_idx;
    reg [CW-1:0] prev_idx;  reg [7:0] consec;  reg [7:0] refr;
    wire signed [23:0] vo_c = vo_flat[c*24 +: 24];
    wire signed [32:0] margin = {best[31], best} - {second[31], second};
    integer i;
    genvar g;
    generate for (g = 0; g < N_CLASS; g = g + 1) begin : G
        assign sum_flat[g*32 +: 32] = sum[g];
    end endgenerate
    assign busy = (st != S_IDLE);
    always @(posedge clk) begin
        done <= 1'b0;
        if (clr) begin
            for (i = 0; i < N_CLASS; i = i + 1) sum[i] <= 32'sd0;
            wp <= 7'd0; win_cnt <= 8'd0; st <= S_IDLE; c <= 0; class_idx <= 0; det_valid <= 1'b0;
            prev_idx <= 0; consec <= 8'd0; refr <= 8'd0;
        end else case (st)
            S_IDLE: if (start) begin c <= 0; st <= S_RD; end
            S_RD: begin                       // issue ring read of the oldest frame, class c
                ring_q <= ring[old_p*N_CLASS + c];
                st <= S_UPD;
            end
            S_UPD: begin
                sum[c] <= sum[c] + {{8{vo_c[23]}}, vo_c} - (sub_ok ? {{8{ring_q[23]}}, ring_q} : 32'sd0);
                ring[wp*N_CLASS + c] <= vo_c;
                if (c == N_CLASS-1) begin c <= 0; st <= S_ARG; end
                else begin c <= c + 1; st <= S_RD; end
            end
            S_ARG: begin                      // sequential argmax, strict '>' keeps the lowest index on ties
                if (c == 0) begin best <= sum[0]; best_idx <= 0; second <= -32'sd2147483648; end
                else if (sum[c] > best) begin second <= best; best <= sum[c]; best_idx <= c; end
                else if (sum[c] > second) second <= sum[c];
                if (c == N_CLASS-1) begin c <= 0; st <= S_FIN; end else c <= c + 1;
            end
            S_FIN: begin
                class_idx <= best_idx;
                wp <= wp + 1;
                if (win_cnt < WIN_SUM) win_cnt <= win_cnt + 1;
                if (best_idx == prev_idx) begin if (consec != 8'hff) consec <= consec + 1; end
                else consec <= 8'd1;
                prev_idx <= best_idx;
                if (refr != 0) refr <= refr - 1;
                det_valid <= (best_idx < N_KEYWORD) && (margin >= MARGIN) && (refr == 0) &&
                             (((best_idx == prev_idx) ? consec + 1 : 8'd1) >= N_CONSEC);
                if ((best_idx < N_KEYWORD) && (margin >= MARGIN) && (refr == 0) &&
                    (((best_idx == prev_idx) ? consec + 1 : 8'd1) >= N_CONSEC) && (REFRACT != 0)) refr <= REFRACT;
                done <= 1'b1; st <= S_IDLE;
            end
            default: st <= S_IDLE;
        endcase
    end
endmodule
