// fe_power.v — block 5: P = re^2 + im^2 (18x18 signed squares -> 36-bit unsigned, exact, no truncation).
`timescale 1ns/1ps
module fe_power #(parameter DB = 18) (
    input  wire signed [DB-1:0] re, input wire signed [DB-1:0] im,
    output wire [2*DB-1:0] p
);
    wire signed [2*DB-1:0] rr = re * re, ii = im * im;   // each <= 2^34 (|x| <= 2^17)
    assign p = $unsigned(rr) + $unsigned(ii);
endmodule
