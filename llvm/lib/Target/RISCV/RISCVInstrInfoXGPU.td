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

def CSR_WTID : SysReg<"wtid", 0xCC0>;
def CSR_LTID : SysReg<"ltid", 0xCC1>;
def CSR_GTID : SysReg<"gtid", 0xCC2>;
def CSR_LWID : SysReg<"lwid", 0xCC3>;
def CSR_GCID : SysReg<"gcid", 0xCC4>;
def CSR_TMASK : SysReg<"tmask", 0xCC5>;

def CSR_RASTPOS : SysReg<"rastpos", 0xCD0>;
def CSR_RASTPID : SysReg<"rastpid", 0xCD1>;
def CSR_RASTBCA : SysReg<"rastbca", 0xCD2>;
def CSR_RASTBCB : SysReg<"rastbcb", 0xCD3>;
def CSR_RASTBCC : SysReg<"rastbcc", 0xCD4>;
def CSR_RASTMASK : SysReg<"rastmask", 0xCD5>;

def CSR_MNT : SysReg<"mnt", 0xFC0>;
def CSR_MNW : SysReg<"mnw", 0xFC1>;
def CSR_MNC : SysReg<"mnc", 0xFC2>;

def : Pat<(int_riscv_gpu_tmask), (CSRRS CSR_TMASK.Encoding, X0)>;

def : Pat<(int_riscv_gpu_tmc GPR:$rs2), (GPU_TMC GPR:$rs2)>;

def : Pat<(int_riscv_gpu_pred GPR:$rs2), (GPU_PRED GPR:$rs2)>;

def : Pat<(int_riscv_gpu_split GPR:$rs2), (GPU_SPLIT GPR:$rs2)>;

def : Pat<(int_riscv_gpu_join), (GPU_JOIN)>;
