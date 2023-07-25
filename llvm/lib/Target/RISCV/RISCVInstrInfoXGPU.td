def OPC_GPU_CTRL : RISCVOpcode<"GPU_CTRL", 0b1101011>;
def OPC_GPU      : RISCVOpcode<"GPU", 0b1011011>;

let hasSideEffects = 1, mayStore = 1 , mayLoad = 1 in {

def GPU_TMC : RVInstS<0, OPC_GPU_CTRL, (outs), (ins GPR:$rs2), "gpu_tmc", "$rs2">, Sched<[]> {
    let imm12 = 0;
    let rs1   = 0;
}

def GPU_WSPAWN : RVInstS<1, OPC_GPU_CTRL, (outs), (ins GPR:$rs1, GPR:$rs2), "gpu_wspawn", "$rs1, $rs2">, Sched<[]> {
    let imm12 = 0;
}

def GPU_SPLIT : RVInstS<2, OPC_GPU_CTRL, (outs), (ins GPR:$rs2), "gpu_split", "$rs2">, Sched<[]> {
    let imm12 = 0;
    let rs1   = 0;
}

def GPU_JOIN : RVInstS<3, OPC_GPU_CTRL, (outs), (ins), "gpu_join", "">, Sched<[]> {
    let imm12 = 0;
    let rs1   = 0;
    let rs2   = 0;    
}

def GPU_BARRIER : RVInstS<4, OPC_GPU_CTRL, (outs), (ins GPR:$rs1, GPR:$rs2), "gpu_barrier", "$rs1, $rs2">, Sched<[]> {
    let imm12 = 0;
}

def GPU_PRED : RVInstS<5, OPC_GPU_CTRL, (outs), (ins GPR:$rs2), "gpu_pred", "$rs2">, Sched<[]> {
    let imm12 = 0;
    let rs1   = 0;
}

def GPU_RAST : RVInstR<0, 0, OPC_GPU, (outs GPR:$rd), (ins), "gpu_rast", "">, Sched<[]> {
    let rs1 = 0;
    let rs2 = 0;
}

}

def : Pat<(int_riscv_gpu_tmc GPR:$rs2), (GPU_TMC GPR:$rs2)>;

def : Pat<(int_riscv_gpu_pred GPR:$rs2), (GPU_PRED GPR:$rs2)>;

def : Pat<(int_riscv_gpu_split GPR:$rs2), (GPU_SPLIT GPR:$rs2)>;

def : Pat<(int_riscv_gpu_join), (GPU_JOIN)>;
