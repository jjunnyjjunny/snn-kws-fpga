// lif_update.v — one LIF neuron update (combinational), exactly snn_int_ref.lif_step:
//   I   = (is_input ? acc >>> IN_SHIFT : acc) + b
//   tmp = v - (v >>> LEAK_K) + I ;  v1 = sat16(tmp) ;  s = (v1 >= theta) ;  v2 = v1 - s*theta
`timescale 1ns/1ps
module lif_update #(
    parameter ACC_W    = 24,
    parameter IN_SHIFT = 6,
    parameter LEAK_K   = 3
) (
    input  wire                    is_input,
    input  wire signed [ACC_W-1:0] acc,
    input  wire signed [15:0]      b,
    input  wire        [15:0]      theta,
    input  wire signed [15:0]      v,
    output wire signed [15:0]      v_next,
    output wire                    spike,
    output wire signed [ACC_W-1:0] cur       // debug: I
);
    wire signed [ACC_W-1:0] acc_sh = is_input ? (acc >>> IN_SHIFT) : acc;
    wire signed [ACC_W-1:0] I      = acc_sh + {{(ACC_W-16){b[15]}}, b};
    wire signed [15:0]      leak   = v >>> LEAK_K;
    wire signed [ACC_W+2:0] tmp    = {{3{v[15]}}, v} - {{3{leak[15]}}, leak} + {{3{I[ACC_W-1]}}, I};
    wire signed [15:0] v1 = (tmp > $signed({{(ACC_W+3-16){1'b0}}, 16'h7fff})) ? 16'sh7fff :
                            (tmp < -$signed({{(ACC_W+3-16){1'b0}}, 16'h8000})) ? -16'sh8000 : tmp[15:0];
    wire signed [16:0] theta_s = {1'b0, theta};
    wire s = ($signed({v1[15], v1}) >= theta_s);
    assign spike  = s;
    assign v_next = s ? (v1 - theta[15:0]) : v1;     // v1 >= theta >= 0 -> no wrap
    assign cur    = I;
endmodule
