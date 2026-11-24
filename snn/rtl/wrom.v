// wrom.v — synchronous-read weight ROM initialised with $readmemh (BRAM inferred).
// Read enable `en` is asserted only in cycles that actually read (BRAM enable gating).
`timescale 1ns/1ps
module wrom #(
    parameter WIDTH   = 8,
    parameter DEPTH   = 7680,
    parameter AW      = 14,
    parameter MEMFILE = "weights.mem"
) (
    input  wire            clk,
    input  wire            en,
    input  wire [AW-1:0]   addr,
    output reg  [WIDTH-1:0] dout
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh(MEMFILE, mem);
    always @(posedge clk) begin
        if (en) dout <= mem[addr];
    end
endmodule
