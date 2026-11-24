// fe_fft512.v — block 4: in-place radix-2 DIT FFT, 512 points, 9 stages, x1/2 per stage, bit-reversed input
// (loaded by fe_window). Butterfly exactly as frontend_ref.fft512_int:
//   p  = b * W  (W = cos - j sin, Q1.15; 18x16 products)          a' = sat18((a<<15 + p + 2^15) >>> 16)
//   b' = sat18((a<<15 - p + 2^15) >>> 16)
// MULT_SERIAL = 0 : 4 multipliers, one butterfly issued every 2 cycles (reads on buffer cycles 1,3,5..,
//                   writes on 6,8,10.. -> no port conflict);   = 1 : one multiplier re-used over 4 cycles,
//                   one butterfly every 4 cycles (reads 1,5,9.., writes 10,14,..).
// Pipeline is drained at every stage boundary (in-place hazard safety).
`timescale 1ns/1ps
module fe_fft512 #(parameter DB = 18, N = 512, LOGN = 9, AW = 9, MULT_SERIAL = 0,
                   TW_RE_MEM = "../rom/twiddle_re.mem", TW_IM_MEM = "../rom/twiddle_im.mem") (
    input  wire clk, input wire rst,
    input  wire start,
    output reg  done,
    output wire busy,
    output reg           en_a, we_a, output reg [AW-1:0] addr_a, output reg [2*DB-1:0] wd_a, input wire [2*DB-1:0] rd_a,
    output reg           en_b, we_b, output reg [AW-1:0] addr_b, output reg [2*DB-1:0] wd_b, input wire [2*DB-1:0] rd_b
);
    localparam SLOT = MULT_SERIAL ? 4 : 2;
    reg signed [15:0] tw_re [0:N/2-1]; reg signed [15:0] tw_im [0:N/2-1];
    initial begin $readmemh(TW_RE_MEM, tw_re); $readmemh(TW_IM_MEM, tw_im); end
    reg run, stage_wait; reg [3:0] stage; reg [7:0] bf; reg [2:0] slot;
    wire [AW-1:0] half = 1 << (stage - 1);
    wire [AW-1:0] k    = bf & (half - 1);
    wire [AW-1:0] ia   = ((bf >> (stage - 1)) << stage) + k;
    wire [AW-1:0] ib   = ia + half;
    wire [7:0]    twi  = k << (LOGN - stage);
    reg        q1_v; reg [AW-1:0] q1_ia, q1_ib; reg signed [15:0] q1_wr, q1_wi;
    reg        q2_v; reg [AW-1:0] q2_ia, q2_ib; reg signed [15:0] q2_wr, q2_wi;
    reg        q3_v; reg [AW-1:0] q3_ia, q3_ib; reg signed [DB-1:0] q3_ar, q3_ai; reg signed [DB+16:0] q3_pr, q3_pi;
    reg        q4_v; reg [AW-1:0] q4_ia, q4_ib; reg [2*DB-1:0] q4_a, q4_b;
    reg        q5_v; reg [AW-1:0] q5_ia, q5_ib; reg [2*DB-1:0] q5_a, q5_b;
    wire signed [DB-1:0] ar = rd_a[2*DB-1:DB], ai = rd_a[DB-1:0], br = rd_b[2*DB-1:DB], bi = rd_b[DB-1:0];
    // parallel products
    wire signed [DB+15:0] m_rr = br * q2_wr, m_ii = bi * q2_wi, m_ri = br * q2_wi, m_ir = bi * q2_wr;
    wire signed [DB+16:0] pr_par = m_rr - m_ii, pi_par = m_ri + m_ir;
    // serial products (one multiplier)
    reg s_act; reg [1:0] ms; reg signed [DB-1:0] s_br, s_bi, s_ar, s_ai; reg signed [15:0] s_wr, s_wi; reg [AW-1:0] s_ia, s_ib;
    reg signed [DB+16:0] s_pr, s_pi;
    wire signed [DB-1:0] s_x = (ms == 2'd0 || ms == 2'd2) ? s_br : s_bi;
    wire signed [15:0]   s_w = (ms == 2'd0 || ms == 2'd3) ? s_wr : s_wi;
    wire signed [DB+15:0] s_m = s_x * s_w;        // ms0: br*wr, ms1: bi*wi, ms2: br*wi, ms3: bi*wr
    function signed [DB-1:0] rsat; input signed [DB+17:0] x; reg signed [DB+17:0] r; begin
        r = (x + $signed({{(DB+2){1'b0}}, 16'h8000})) >>> 16;
        if (r > $signed({{18{1'b0}}, {1'b0, {(DB-1){1'b1}}}})) rsat = {1'b0, {(DB-1){1'b1}}};
        else if (r < -$signed({{18{1'b0}}, {1'b1, {(DB-1){1'b0}}}})) rsat = {1'b1, {(DB-1){1'b0}}};
        else rsat = r[DB-1:0];
    end endfunction
    wire signed [DB+17:0] a15r = {{3{q3_ar[DB-1]}}, q3_ar, 15'd0}, a15i = {{3{q3_ai[DB-1]}}, q3_ai, 15'd0};   // a<<15, sign-extended
    wire signed [DB+17:0] p3r = {q3_pr[DB+16], q3_pr}, p3i = {q3_pi[DB+16], q3_pi};
    wire signed [DB-1:0] nar = rsat(a15r + p3r), nai = rsat(a15i + p3i), nbr = rsat(a15r - p3r), nbi = rsat(a15i - p3i);
    wire pipe_busy = q1_v | q2_v | q3_v | q4_v | q5_v | s_act;
`ifdef ACTIVITY
    integer act_mult = 0, act_cyc = 0;
    always @(posedge clk) begin if (MULT_SERIAL == 0 && q2_v) act_mult = act_mult + 4; if (MULT_SERIAL != 0 && s_act) act_mult = act_mult + 1; if (busy) act_cyc = act_cyc + 1; end
`endif
    assign busy = run | stage_wait | pipe_busy;
    always @(posedge clk) begin
        done <= 1'b0; en_a <= 0; we_a <= 0; en_b <= 0; we_b <= 0;
        q1_v <= 0; q3_v <= 0;
        if (rst) begin run <= 0; stage_wait <= 0; stage <= 1; bf <= 0; slot <= 0; q2_v <= 0; q4_v <= 0; q5_v <= 0; s_act <= 0; ms <= 0; end
        else begin
            if (start) begin run <= 1'b1; stage_wait <= 0; stage <= 1; bf <= 0; slot <= 0; end
            // ---- stage boundary: wait until the pipeline is empty, then next stage (or finish)
            if (stage_wait && !pipe_busy) begin
                stage_wait <= 0;
                if (stage == LOGN) done <= 1'b1; else begin stage <= stage + 1'b1; run <= 1'b1; slot <= 0; end
            end
            // ---- issue reads at slot 0
            if (run && slot == 0) begin
                en_a <= 1; addr_a <= ia; en_b <= 1; addr_b <= ib;
                q1_v <= 1; q1_ia <= ia; q1_ib <= ib; q1_wr <= tw_re[twi]; q1_wi <= tw_im[twi];
                if (bf == 8'd255) begin bf <= 0; run <= 0; stage_wait <= 1; end else bf <= bf + 1'b1;
            end
            if (run) slot <= (slot == SLOT - 1) ? 3'd0 : slot + 1'b1;
            // ---- q2: buffer data valid
            q2_v <= q1_v; q2_ia <= q1_ia; q2_ib <= q1_ib; q2_wr <= q1_wr; q2_wi <= q1_wi;
            if (MULT_SERIAL == 0) begin
                if (q2_v) begin q3_v <= 1; q3_ia <= q2_ia; q3_ib <= q2_ib; q3_ar <= ar; q3_ai <= ai; q3_pr <= pr_par; q3_pi <= pi_par; end
            end else begin
                if (q2_v) begin s_act <= 1; ms <= 0; s_br <= br; s_bi <= bi; s_wr <= q2_wr; s_wi <= q2_wi; s_ar <= ar; s_ai <= ai; s_ia <= q2_ia; s_ib <= q2_ib; end
                if (s_act) begin
                    ms <= ms + 1'b1;
                    case (ms)
                        2'd0: s_pr <= {s_m[DB+15], s_m};
                        2'd1: s_pr <= s_pr - {s_m[DB+15], s_m};
                        2'd2: s_pi <= {s_m[DB+15], s_m};
                        2'd3: begin q3_v <= 1; q3_ia <= s_ia; q3_ib <= s_ib; q3_ar <= s_ar; q3_ai <= s_ai; q3_pr <= s_pr; q3_pi <= s_pi + {s_m[DB+15], s_m};
                                    if (!q2_v) s_act <= 0; end
                    endcase
                end
            end
            // ---- q4: results ; q5: write issue (lands on a buffer cycle without reads)
            q4_v <= q3_v; q4_ia <= q3_ia; q4_ib <= q3_ib; q4_a <= {nar, nai}; q4_b <= {nbr, nbi};
            q5_v <= q4_v; q5_ia <= q4_ia; q5_ib <= q4_ib; q5_a <= q4_a; q5_b <= q4_b;
            if (q5_v) begin en_a <= 1; we_a <= 1; addr_a <= q5_ia; wd_a <= q5_a; en_b <= 1; we_b <= 1; addr_b <= q5_ib; wd_b <= q5_b; end
        end
    end
endmodule
