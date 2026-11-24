// row_acc.v — current accumulation engine shared by all layers.
// For every issued (post, act) the weight arrives from the ROM one cycle later; the product
// (act x w for the input layer, w alone for spike inputs — no multiply) is added to acc[post]
// one cycle after that. acc[] is 64 x 24-bit registers; `clr` zeroes them at layer start.
// Rows with activation 0 are never issued (zero-skip / event-driven), so no cycle is spent on them.
`timescale 1ns/1ps
module row_acc #(
    parameter N_POST = 64,
    parameter PW     = 6,
    parameter ACC_W  = 24
) (
    input  wire              clk,
    input  wire              clr,
    input  wire              issue,      // address issued to the ROM this cycle
    input  wire              is_input,   // input layer: multiply by act
    input  wire [7:0]        act,        // activation of the issued row (0..255 or 1)
    input  wire [PW-1:0]     post,       // post-synaptic index of the issued element
    input  wire signed [7:0] rom_dout,   // weight, valid one cycle after issue
    input  wire [PW-1:0]     rd_post,
    output wire signed [ACC_W-1:0] rd_acc,
    output wire              busy        // pipeline non-empty
);
    reg signed [ACC_W-1:0] acc [0:N_POST-1];
    // stage 1: ROM data valid
    reg              p1_v, p1_in;
    reg [7:0]        p1_act;
    reg [PW-1:0]     p1_post;
    // stage 2: product ready
    reg              p2_v;
    reg [PW-1:0]     p2_post;
    reg signed [ACC_W-1:0] p2_prod;
    wire signed [16:0] prod_mul = $signed({1'b0, p1_act}) * rom_dout;   // 9b x 8b signed
    wire signed [ACC_W-1:0] prod = p1_in ? {{(ACC_W-17){prod_mul[16]}}, prod_mul}
                                         : {{(ACC_W-8){rom_dout[7]}}, rom_dout};
    integer i;
    always @(posedge clk) begin
        p1_v <= issue; p1_in <= is_input; p1_act <= act; p1_post <= post;
        p2_v <= p1_v;  p2_post <= p1_post; p2_prod <= prod;
        if (clr) begin
            for (i = 0; i < N_POST; i = i + 1) acc[i] <= {ACC_W{1'b0}};
        end else if (p2_v) begin
            acc[p2_post] <= acc[p2_post] + p2_prod;
        end
    end
    assign rd_acc = acc[rd_post];
    assign busy   = p1_v | p2_v;
endmodule
