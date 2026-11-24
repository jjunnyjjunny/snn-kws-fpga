// prom.v — small asynchronous-read parameter ROM (theta / bias, 64 x 16) -> distributed LUT ROM.
`timescale 1ns/1ps
module prom #(
    parameter WIDTH   = 16,
    parameter DEPTH   = 64,
    parameter AW      = 6,
    parameter MEMFILE = "params.mem"
) (
    input  wire [AW-1:0]    addr,
    output wire [WIDTH-1:0] dout
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    initial $readmemh(MEMFILE, mem);
    assign dout = mem[addr];
endmodule
