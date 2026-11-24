// fe_mel_log.v — blocks 6-8: sparse mel filterbank on the FFT buffer bins, log2 (leading one + 5-bit mantissa
// LUT, Q6.4), feature = sat8(log2q - LOG_OFFSET). Also emits the frame energies used by the VAD:
//   e_sum  = sum of the 40 features (14 bit)            e_log = log2q of the sum of the 40 mel accumulators.
// Per band: acc = sum_j w[m][j] * P[start+j]  (50-bit unsigned accumulator, 36x8 multiply);
// per bin the buffer is read (re, im) and the power computed on the fly (fe_power) -> no power buffer.
// Output stream: feat_valid / feat_data / feat_last (one feature per cycle), then done.
// ROMs: mel40_table.mem {start[25:17], len[16:10], woff[9:0]}, mel40_weights.mem (8-bit), log2_lut.mem (5-bit).
`timescale 1ns/1ps
module fe_mel_log #(parameter DB = 18, AW = 9, N_MELS = 40, ACC_W = 50, LOG_OFFSET = 332, N_WEIGHTS = 488,
                    TABLE_MEM = "../rom/mel40_table.mem", WEIGHT_MEM = "../rom/mel40_weights.mem", LUT_MEM = "../rom/log2_lut.mem") (
    input  wire clk, input wire rst,
    input  wire start,
    output reg  done,
    output wire busy,
    output reg           en_a, output reg [AW-1:0] addr_a, input wire [2*DB-1:0] rd_a,     // buffer read port
    output reg           feat_valid, output reg [7:0] feat_data, output reg feat_last,
    output reg  [13:0]   e_sum,
    output reg  [9:0]    e_log,
    // debug taps (test bench): per-band accumulator and log2q as they are produced
    output reg           dbg_valid, output reg [ACC_W-1:0] dbg_acc, output reg [9:0] dbg_log2q
);
    reg [25:0] table_rom [0:N_MELS-1]; reg [7:0] w_rom [0:N_WEIGHTS-1]; reg [4:0] lut [0:31];
    initial begin $readmemh(TABLE_MEM, table_rom); $readmemh(WEIGHT_MEM, w_rom); $readmemh(LUT_MEM, lut); end
    localparam S_IDLE=0, S_BAND=1, S_RUN=2, S_DRAIN=3, S_LOG=4, S_EMIT=5, S_ELOG=6;
    reg [2:0] st; reg [5:0] band; reg [6:0] j, len; reg [8:0] bstart; reg [9:0] woff;
    reg [ACC_W-1:0] acc; reg [ACC_W+5:0] acc_sum;        // sum over bands (56 bit)
    reg [7:0] feat [0:N_MELS-1]; reg [13:0] fsum;
    // pipeline: issue read (p1) -> data (p2) -> power*w (p3) -> accumulate
    reg p1_v, p2_v, p3_v; reg [7:0] p1_w, p2_w; reg [ACC_W-1:0] p3_prod;
    wire [2*DB-1:0] pw;
    fe_power #(.DB(DB)) u_pow (.re(rd_a[2*DB-1:DB]), .im(rd_a[DB-1:0]), .p(pw));
    wire [2*DB+7:0] prod = pw * p2_w;                      // 36 x 8 -> 44 bit
    // log2: leading-one position + 5-bit mantissa
    function [9:0] log2q; input [ACC_W+5:0] x; integer i; reg [5:0] pos; reg [4:0] mant; reg [ACC_W+5:0] sh; begin
        pos = 0; for (i = 0; i < ACC_W+6; i = i + 1) if (x[i]) pos = i;
        sh = (pos >= 5) ? (x >> (pos - 5)) : (x << (5 - pos));
        mant = sh[4:0];
        log2q = (x == 0) ? 10'd0 : ({4'd0, pos} << 4) + {5'd0, lut[mant]};
    end endfunction
    wire [9:0] lq = log2q({6'd0, acc});
    wire signed [11:0] fdiff = $signed({2'b0, lq}) - LOG_OFFSET;
    wire [7:0] f8 = (fdiff < 0) ? 8'd0 : (fdiff > 255) ? 8'd255 : fdiff[7:0];
    assign busy = (st != S_IDLE);
`ifdef ACTIVITY
    integer act_mult = 0, act_cyc = 0;      // per bin: 2 squares (power) + 1 (P*w)
    always @(posedge clk) begin if (p2_v) act_mult = act_mult + 3; if (busy) act_cyc = act_cyc + 1; end
`endif
    integer i;
    always @(posedge clk) begin
        done <= 0; en_a <= 0; feat_valid <= 0; feat_last <= 0; dbg_valid <= 0;
        p1_v <= 0;
        if (rst) begin st <= S_IDLE; p2_v <= 0; p3_v <= 0; end
        else begin
            // ---- bin pipeline (independent of state so it drains naturally)
            p2_v <= p1_v; p2_w <= p1_w;
            p3_v <= p2_v; p3_prod <= {{(ACC_W-2*DB-8){1'b0}}, prod};
            if (p3_v) acc <= acc + p3_prod;
            case (st)
            S_IDLE: if (start) begin st <= S_BAND; band <= 0; acc_sum <= 0; fsum <= 0; end
            S_BAND: begin                                             // fetch band descriptor
                bstart <= table_rom[band][25:17]; len <= table_rom[band][16:10]; woff <= table_rom[band][9:0];
                j <= 0; acc <= 0; st <= S_RUN;
            end
            S_RUN: begin                                              // issue one bin read per cycle
                en_a <= 1; addr_a <= bstart + j; p1_v <= 1; p1_w <= w_rom[woff + j];
                if (j == len - 1) st <= S_DRAIN; else j <= j + 1'b1;
            end
            S_DRAIN: if (!p1_v && !p2_v && !p3_v) st <= S_LOG;      // wait for the last accumulate
            S_LOG: begin
                feat[band] <= f8; fsum <= fsum + f8; acc_sum <= acc_sum + acc;
                dbg_valid <= 1; dbg_acc <= acc; dbg_log2q <= lq;
                if (band == N_MELS - 1) begin st <= S_ELOG; end else begin band <= band + 1'b1; st <= S_BAND; end
            end
            S_ELOG: begin e_sum <= fsum; e_log <= log2q(acc_sum); band <= 0; st <= S_EMIT; end
            S_EMIT: begin
                feat_valid <= 1; feat_data <= feat[band]; feat_last <= (band == N_MELS - 1);
                if (band == N_MELS - 1) begin st <= S_IDLE; done <= 1; end else band <= band + 1'b1;
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
