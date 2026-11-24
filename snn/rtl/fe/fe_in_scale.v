// fe_in_scale.v — block 1: x18 = sat18((pcm24 + 2^5) >>> 6)   (frontend_ref.scale_input, round-half-up)
`timescale 1ns/1ps
module fe_in_scale #(parameter IN_SHIFT = 6, DB = 18) (
    input  wire signed [23:0] pcm,
    output wire signed [DB-1:0] x
);
    wire signed [24:0] r = {pcm[23], pcm} + (25'sd1 <<< (IN_SHIFT - 1));
    wire signed [24:0] s = r >>> IN_SHIFT;
    assign x = (s > $signed({{(25-DB){1'b0}}, {1'b0, {(DB-1){1'b1}}}})) ? {1'b0, {(DB-1){1'b1}}} :
               (s < -$signed({{(25-DB){1'b0}}, {1'b1, {(DB-1){1'b0}}}})) ? {1'b1, {(DB-1){1'b0}}} : s[DB-1:0];
endmodule
