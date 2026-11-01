// OpenQASM 2.0 C++ 解析器的 gtest：寄存器拼接与广播、qelib1 门的导入与按定义展开、
// 带形参的用户门、表达式求值（与 Python float 语义一致的幂运算等），以及参考实现会拒绝
// 的各种输入一律返回 false（调用方据此回退到 Python 参考实现）。
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/ir/circuit.hpp"
#include "quspect_native/ir/qasm2_parser.hpp"

namespace {

using quspect::native::ir::Circuit;
using quspect::native::ir::CompileQasm2Library;
using quspect::native::ir::OpKind;
using quspect::native::ir::ParseQasm2;
using quspect::native::ir::Qasm2Library;

// Python 端 _QELIB1_DEFINITIONS 的一部分（足以覆盖展开逻辑）。
constexpr const char* kLibrary = R"(
gate u0(gamma) q { id q; }
gate cy a,b { sdg b; cx a,b; s b; }
gate cswap a,b,c { cx c,b; ccx a,b,c; cx c,b; }
gate cu(theta,phi,lambda,gamma) c,t { p(gamma) c; cu3(theta,phi,lambda) c,t; }
gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }
)";

const Qasm2Library& Lib() {
    static auto lib = CompileQasm2Library(kLibrary);
    return *lib;
}

Circuit Parse(const std::string& body, bool expect_ok = true) {
    Circuit c;
    std::string err;
    const bool ok = ParseQasm2("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" + body, Lib(), c, &err);
    EXPECT_EQ(ok, expect_ok) << err << "\n" << body;
    return c;
}

bool Accepts(const std::string& program) {
    Circuit c;
    return ParseQasm2(program, Lib(), c);
}

TEST(Qasm2Parser, RegistersBroadcastAndMeasure) {
    Circuit c = Parse("qreg a[2]; qreg b[2]; creg c[2];\nh a; cx a,b; cx a[0],b; measure b -> c;\n");
    // h a -> 2, cx a,b -> 2, cx a[0],b -> 2, measure -> 2
    ASSERT_EQ(c.size(), 8u);
    EXPECT_EQ(c[0].op.kind, OpKind::kH);
    EXPECT_EQ(c[1].qubits[0], 1u);
    EXPECT_EQ(c[2].op.kind, OpKind::kCX);
    EXPECT_EQ(c[2].qubits[0], 0u);
    EXPECT_EQ(c[2].qubits[1], 2u);
    EXPECT_EQ(c[5].qubits[1], 3u);
    EXPECT_EQ(c[6].op.kind, OpKind::kMeasure);
    EXPECT_EQ(c[7].qubits[0], 3u);
    EXPECT_EQ(c[7].clbits[0], 1u);
}

TEST(Qasm2Parser, NativeAliasesAndParameters) {
    Circuit c = Parse("qreg q[2];\nu1(pi/4) q[0]; cu1(-pi/2) q[0],q[1]; U(1,2,3) q[1]; CX q[0],q[1]; u(0.5,.5,5.) q[0];\n");
    ASSERT_EQ(c.size(), 5u);
    EXPECT_EQ(c[0].op.kind, OpKind::kP);
    EXPECT_EQ(c[0].op.params[0], 3.141592653589793 / 4);
    EXPECT_EQ(c[1].op.kind, OpKind::kCP);
    EXPECT_EQ(c[1].op.params[0], -(3.141592653589793 / 2));
    EXPECT_EQ(c[2].op.kind, OpKind::kU3);
    EXPECT_EQ(c[2].op.num_params, 3);
    EXPECT_EQ(c[3].op.kind, OpKind::kCX);
    EXPECT_EQ(c[4].op.params[2], 5.0);
}

TEST(Qasm2Parser, ExpressionSemantics) {
    Circuit c = Parse("qreg q[1];\nrz(2^3^2) q[0]; rz(-2^2) q[0]; rz((-2)^3) q[0]; rz(sqrt(2)*ln(3)) q[0];"
                      " rz(1e-3+2*3-4/8) q[0]; rz(0^0) q[0]; rz(--+1) q[0];\n");
    ASSERT_EQ(c.size(), 7u);
    EXPECT_EQ(c[0].op.params[0], 512.0);  // 右结合
    EXPECT_EQ(c[1].op.params[0], -4.0);   // 一元负号作用于幂
    EXPECT_EQ(c[2].op.params[0], -8.0);
    EXPECT_EQ(c[3].op.params[0], std::sqrt(2.0) * std::log(3.0));
    EXPECT_EQ(c[4].op.params[0], 1e-3 + 2.0 * 3.0 - 4.0 / 8.0);
    EXPECT_EQ(c[5].op.params[0], 1.0);
    EXPECT_EQ(c[6].op.params[0], 1.0);
}

TEST(Qasm2Parser, LibraryAndUserGatesExpand) {
    Circuit c = Parse("qreg q[3];\ngate g(a,b) x,y { rzz(a*b) x,y; barrier x; cy y,x; }\n"
                      "g(2,0.25) q[2],q[0]; cswap q[0],q[1],q[2];\n");
    // rzz -> cx rz cx ; cy -> sdg cx s ; cswap -> cx ccx cx
    ASSERT_EQ(c.size(), 9u);
    EXPECT_EQ(c[0].op.kind, OpKind::kCX);
    EXPECT_EQ(c[0].qubits[0], 2u);
    EXPECT_EQ(c[1].op.kind, OpKind::kRZ);
    EXPECT_EQ(c[1].op.params[0], 0.5);
    EXPECT_EQ(c[3].op.kind, OpKind::kSdg);
    EXPECT_EQ(c[3].qubits[0], 2u);
    EXPECT_EQ(c[7].op.kind, OpKind::kCCX);
}

TEST(Qasm2Parser, BareBarrierUsesActiveQubitsSorted) {
    Circuit c = Parse("qreg q[4];\nx q[2]; h q[0]; barrier; barrier q[3],q,q[1];\n");
    ASSERT_EQ(c.size(), 4u);
    EXPECT_EQ(c[2].op.kind, OpKind::kBarrier);
    ASSERT_EQ(c[2].qubits.size(), 2u);
    EXPECT_EQ(c[2].qubits[0], 0u);
    EXPECT_EQ(c[2].qubits[1], 2u);
    ASSERT_EQ(c[3].qubits.size(), 4u);
    EXPECT_EQ(c[3].qubits[0], 3u);
    EXPECT_EQ(c[3].qubits[1], 0u);
}

TEST(Qasm2Parser, ShadowingLibraryGateWithoutInclude) {
    Circuit c;
    ASSERT_TRUE(ParseQasm2("OPENQASM 2.0; qreg q[1]; gate h a { x a; } h q[0];", Lib(), c));
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0].op.kind, OpKind::kX);
    // 包含 qelib1.inc 之后不能重定义
    EXPECT_FALSE(Accepts("OPENQASM 2.0; include \"qelib1.inc\"; gate h a { x a; }"));
    // U / CX 永远不能重定义
    EXPECT_FALSE(Accepts("OPENQASM 2.0; gate U(a,b,c) q { x q; }"));
}

TEST(Qasm2Parser, RejectsWhatTheReferenceRejects) {
    const char* bad[] = {
        "OPENQASM 3.0; qreg q[1];",
        "OPENQASM 2.0; qreg q[1]; reset q[0];",
        "OPENQASM 2.0; qreg q[1]; creg c[1]; if (c == 1) x q[0];",
        "OPENQASM 2.0; qreg q[1]; foo q[0];",
        "OPENQASM 2.0; qreg q[1]; x q[1];",
        "OPENQASM 2.0; qreg q[0];",
        "OPENQASM 2.0; qreg q[1]; qreg q[2];",
        "OPENQASM 2.0; qreg q[2]; cx q[0],q[0];",
        "OPENQASM 2.0; qreg a[2]; qreg b[3]; cx a,b;",
        "OPENQASM 2.0; qreg q[1]; rz(1/0) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz(0^-1) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz((-8)^0.5) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz(ln(0)) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz(sqrt(-1)) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz(theta) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz(1,2) q[0];",
        "OPENQASM 2.0; qreg q[1]; rz q[0];",
        "OPENQASM 2.0; qreg q[1]; opaque o a; o q[0];",
        "OPENQASM 2.0; qreg q[1]; creg c[2]; measure q -> c;",
        "OPENQASM 2.0; qreg q[1]; gate g a { measure a; }",
        "OPENQASM 2.0; qreg q[1]; gate g a { y b; }",
        "OPENQASM 2.0; qreg q[1]; gate g(a,a) b { x b; }",
        "OPENQASM 2.0; qreg q[1]; x q[0]",
        "OPENQASM 2.0; qreg q[1]; x q[0]; $",
        "OPENQASM 2.0; qreg q[1]; x q[0]; OPENQASM 2.0;",
        "OPENQASM 2.0; include \"qelib1.inc; qreg q[1];",
        "OPENQASM 2.0; qreg q[1]; rz(.) q[0];",
    };
    for (const char* program : bad) {
        EXPECT_FALSE(Accepts(program)) << program;
    }
    // 没有 include 也能用库门（库门始终可用，只有重定义规则受 include 影响）
    EXPECT_TRUE(Accepts("OPENQASM 2.0; qreg q[2]; cy q[0],q[1]; // comment\n"));
    EXPECT_TRUE(Accepts("OPENQASM 2; ; qreg q[1]; include \"other.inc\"; rz(1e2) q[0];"));
}

TEST(Qasm2Parser, LibraryCompileErrorsThrow) {
    EXPECT_THROW(CompileQasm2Library("gate broken a { nope a; }"), std::invalid_argument);
}

}  // namespace
