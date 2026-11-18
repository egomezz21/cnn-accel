`default_nettype none

module mac(

    input logic     clk,
    input logic     rst,
    input logic     clear,
    input logic     en,
    input logic signed [7:0] a,
    input logic signed [7:0] b,

    output logic signed [31:0] acc
);

 logic signed [15:0] product;
 logic signed [31:0] next_acc;

 always_comb begin
     product = a*b;
     next_acc = acc + 32'(product);
 end

 always_ff @(posedge clk) begin
    if(rst) begin
        acc <= 0;
    end else if (clear) begin
        acc <= 0;
    end else if (en) begin
        acc <= next_acc;
    end
 end

endmodule
