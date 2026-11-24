// vad.v — frame-level voice activity state machine (bit-exact with snn/eval/vad_eval.py VAD.step):
//   idle  : cnt_on = (E > T_ON) ? cnt_on+1 : 0 ; cnt_on >= N_ON -> start (feed the last P frames + this frame)
//   active: n_fed++ ; cnt_off = (E < T_OFF) ? cnt_off+1 : 0 ; n_fed >= L_MAX -> force end ; cnt_off >= H -> end
//   a segment is kept (decision taken) only if n_fed >= L_MIN.
// `strobe` = one pulse per frame with E valid. Outputs are one-cycle pulses in the cycle after the strobe.
`timescale 1ns/1ps
module vad #(parameter EW = 16, T_ON = 600, T_OFF = 360, N_ON = 4, H = 30, P = 20, L_MIN = 25, L_MAX = 130) (
    input  wire clk, input wire rst,
    input  wire strobe, input wire [EW-1:0] E,
    output reg  seg_start,          // pulse: feed `n_pre` buffered frames (incl. current) after resetting the SNN
    output reg  [7:0] n_pre,        // = min(P + 1, frames seen so far)
    output reg  active,
    output reg  seg_end,            // pulse: segment finished after feeding the current frame
    output reg  seg_keep,           // valid with seg_end: n_fed >= L_MIN -> take the decision
    output reg  seg_forced,
    output reg  [7:0] n_fed
);
    reg [7:0] cnt_on, cnt_off, seen;
    always @(posedge clk) begin
        seg_start <= 0; seg_end <= 0;
        if (rst) begin active <= 0; cnt_on <= 0; cnt_off <= 0; n_fed <= 0; seen <= 0; seg_keep <= 0; seg_forced <= 0; end
        else if (strobe) begin
            if (seen != 8'hff) seen <= seen + 1'b1;
            if (!active) begin
                if (E > T_ON) begin
                    if (cnt_on + 1 >= N_ON) begin
                        active <= 1; cnt_on <= 0; cnt_off <= 0; n_fed <= P + 1; seg_start <= 1;
                        n_pre <= (seen + 1 < P + 1) ? seen + 1 : P + 1;
                    end else cnt_on <= cnt_on + 1'b1;
                end else cnt_on <= 0;
            end else begin
                n_fed <= n_fed + 1'b1;
                cnt_off <= (E < T_OFF) ? cnt_off + 1'b1 : 8'd0;
                if (n_fed + 1 >= L_MAX) begin active <= 0; seg_end <= 1; seg_forced <= 1; seg_keep <= (n_fed + 1 >= L_MIN); end
                else if ((E < T_OFF) && (cnt_off + 1 >= H)) begin active <= 0; seg_end <= 1; seg_forced <= 0; seg_keep <= (n_fed + 1 >= L_MIN); end
            end
        end
    end
endmodule
