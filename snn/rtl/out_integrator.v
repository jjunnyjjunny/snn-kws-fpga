// out_integrator.v — one output-class leaky integrator step (combinational):
//   vo' = sat24(vo - (vo >>> LEAK_K) + acc + b)
`timescale 1ns/1ps
module out_integrator #(
    parameter ACC_W  = 24,
    parameter LEAK_K = 3
) (
    input  wire signed [ACC_W-1:0] acc,
    input  wire signed [15:0]      b,
    input  wire signed [23:0]      vo,
    output wire signed [23:0]      vo_next
);
    wire signed [23:0]      leak = vo >>> LEAK_K;
    wire signed [ACC_W-1:0] I    = acc + {{(ACC_W-16){b[15]}}, b};
    wire signed [ACC_W+2:0] tmp  = {{3{vo[23]}}, vo} - {{3{leak[23]}}, leak} + {{3{I[ACC_W-1]}}, I};
    assign vo_next = (tmp > $signed({{(ACC_W+3-24){1'b0}}, 24'h7fffff})) ? 24'sh7fffff :
                     (tmp < -$signed({{(ACC_W+3-24){1'b0}}, 24'h800000})) ? -24'sh800000 : tmp[23:0];
endmodule
