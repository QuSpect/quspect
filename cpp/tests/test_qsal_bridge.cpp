#include <cmath>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/executor/executor.hpp"
#include "quspect_native/ir/program_builder.hpp"
#include "quspect_native/ir/qsal_decoder.hpp"
#include "quspect_native/ir/validation.hpp"
#include "quspect_native/runtime/program.hpp"

using namespace quspect::native;

// ============================================================
// QSAL JSON 样例
// ============================================================

/// 最简 QSAL JSON: 空线路
static const std::string kEmptyCircuit = R"({
  "qsal_version": "0.1.0",
  "name": "empty",
  "circuit": {
    "num_qubits": 2,
    "num_clbits": 0,
    "instructions": []
  }
})";

/// 单 qubit H 门线路
static const std::string kHHGate = R"({
  "qsal_version": "0.1.0",
  "name": "h_gate",
  "circuit": {
    "num_qubits": 1,
    "num_clbits": 0,
    "instructions": [
      {"type": "GATE", "qubits": [0], "clbits": [], "gate": {"name": "h", "num_qubits": 1, "gate_type": "standard"}}
    ]
  }
})";

/// Bell 态线路 + 测量
static const std::string kBellState = R"({
  "qsal_version": "0.1.0",
  "name": "bell",
  "circuit": {
    "num_qubits": 2,
    "num_clbits": 2,
    "instructions": [
      {"type": "GATE", "qubits": [0], "clbits": [], "gate": {"name": "h", "num_qubits": 1, "gate_type": "standard"}},
      {"type": "GATE", "qubits": [0, 1], "clbits": [], "gate": {"name": "cx", "num_qubits": 2, "gate_type": "standard"}},
      {"type": "MEASURE", "qubits": [0, 1], "clbits": [0, 1]}
    ]
  }
})";

/// 参数化门线路 (RX)
static const std::string kRXGate = R"({
  "qsal_version": "0.1.0",
  "name": "rx_gate",
  "circuit": {
    "num_qubits": 1,
    "num_clbits": 0,
    "instructions": [
      {"type": "GATE", "qubits": [0], "clbits": [], "gate": {"name": "rx", "num_qubits": 1, "gate_type": "parameterized", "params": [3.141592653589793]}}
    ]
  }
})";

/// 含 observable 的线路
static const std::string kWithObservable = R"({
  "qsal_version": "0.1.0",
  "name": "obs_test",
  "circuit": {
    "num_qubits": 2,
    "num_clbits": 0,
    "instructions": [
      {"type": "GATE", "qubits": [0], "clbits": [], "gate": {"name": "h", "num_qubits": 1, "gate_type": "standard"}},
      {"type": "GATE", "qubits": [0, 1], "clbits": [], "gate": {"name": "cx", "num_qubits": 2, "gate_type": "standard"}}
    ]
  },
  "observables": [
    {"name": "ZZ", "operator": {"terms": [{"paulis": [[0, "Z"], [1, "Z"]], "coefficient": 1.0}]}},
    {"name": "XX", "operator": {"terms": [{"paulis": [[0, "X"], [1, "X"]], "coefficient": 1.0}]}}
  ]
})";

/// Barrier 线路
static const std::string kBarrierCircuit = R"({
  "qsal_version": "0.1.0",
  "name": "barrier_test",
  "circuit": {
    "num_qubits": 2,
    "num_clbits": 0,
    "instructions": [
      {"type": "GATE", "qubits": [0], "clbits": [], "gate": {"name": "h", "num_qubits": 1, "gate_type": "standard"}},
      {"type": "BARRIER", "qubits": [0, 1]},
      {"type": "GATE", "qubits": [0, 1], "clbits": [], "gate": {"name": "cx", "num_qubits": 2, "gate_type": "standard"}}
    ]
  }
})";

// ============================================================
// QSALDecoder 测试
// ============================================================

TEST(QSALDecoderTest, DecodeEmptyCircuit) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kEmptyCircuit);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    EXPECT_EQ(prog->num_qubits, 2);
    EXPECT_EQ(prog->num_clbits, 0);
    EXPECT_TRUE(prog->ops.empty());
    EXPECT_TRUE(prog->observables.empty());
}

TEST(QSALDecoderTest, DecodeHGate) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kHHGate);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    EXPECT_EQ(prog->num_qubits, 1);
    ASSERT_EQ(prog->ops.size(), 1);

    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[0]));
    const auto& gate = std::get<runtime::GateOperation>(prog->ops[0]);
    EXPECT_EQ(gate.opcode, runtime::OpCode::kNamedGate);
    EXPECT_EQ(gate.name, "h");
    ASSERT_EQ(gate.qubits.size(), 1);
    EXPECT_EQ(gate.qubits[0], 0);
}

TEST(QSALDecoderTest, DecodeBellState) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kBellState);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    EXPECT_EQ(prog->num_qubits, 2);
    EXPECT_EQ(prog->num_clbits, 2);
    ASSERT_EQ(prog->ops.size(), 3);

    // H gate
    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[0]));
    EXPECT_EQ(std::get<runtime::GateOperation>(prog->ops[0]).name, "h");

    // CX gate
    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[1]));
    EXPECT_EQ(std::get<runtime::GateOperation>(prog->ops[1]).name, "cx");

    // Measure
    ASSERT_TRUE(std::holds_alternative<runtime::MeasureOperation>(prog->ops[2]));
    const auto& meas = std::get<runtime::MeasureOperation>(prog->ops[2]);
    ASSERT_EQ(meas.qubits.size(), 2);
    EXPECT_EQ(meas.qubits[0], 0);
    EXPECT_EQ(meas.qubits[1], 1);
    ASSERT_EQ(meas.clbits.size(), 2);
    EXPECT_EQ(meas.clbits[0], 0);
    EXPECT_EQ(meas.clbits[1], 1);
}

TEST(QSALDecoderTest, DecodeRXGate) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kRXGate);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->ops.size(), 1);
    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[0]));
    const auto& gate = std::get<runtime::GateOperation>(prog->ops[0]);
    EXPECT_EQ(gate.name, "rx");
    ASSERT_EQ(gate.params_f64.size(), 1);
    EXPECT_NEAR(gate.params_f64[0], 3.141592653589793, 1e-10);
    ASSERT_EQ(gate.parameter_expressions.size(), 1);
    EXPECT_DOUBLE_EQ(gate.parameter_expressions[0].constant, 3.141592653589793);
    EXPECT_FALSE(gate.parameter_expressions[0].IsSymbolic());
}

TEST(QSALDecoderTest, DecodePhaseGateAsDenseMatrix) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [
          {"type": "GATE", "qubits": [0], "gate": {
            "name": "p", "gate_type": "parameterized", "params": [0.73]
          }}
        ]
      }
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->ops.size(), 1);
    const auto& gate = std::get<runtime::GateOperation>(prog->ops[0]);
    EXPECT_EQ(gate.name, "p");
    EXPECT_EQ(gate.opcode, runtime::OpCode::kDenseGate);
    ASSERT_EQ(gate.dense_matrix_f64.size(), 4);
    EXPECT_NEAR(gate.dense_matrix_f64[0].real(), 1.0, 1e-12);
    EXPECT_NEAR(gate.dense_matrix_f64[3].real(), std::cos(0.73), 1e-12);
    EXPECT_NEAR(gate.dense_matrix_f64[3].imag(), std::sin(0.73), 1e-12);
}

TEST(QSALDecoderTest, DecodeSqrtXGatesAsDenseMatrices) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [
          {"type": "GATE", "qubits": [0], "gate": {"name": "sx", "gate_type": "standard"}},
          {"type": "GATE", "qubits": [0], "gate": {"name": "sxdg", "gate_type": "standard"}}
        ]
      }
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->ops.size(), 2);
    const auto& sx = std::get<runtime::GateOperation>(prog->ops[0]);
    const auto& sxdg = std::get<runtime::GateOperation>(prog->ops[1]);
    EXPECT_EQ(sx.name, "sx");
    EXPECT_EQ(sx.opcode, runtime::OpCode::kDenseGate);
    ASSERT_EQ(sx.dense_matrix_f64.size(), 4);
    ASSERT_EQ(sxdg.dense_matrix_f64.size(), 4);
    // SX = ½[[1+i, 1-i], [1-i, 1+i]] and SXdg is its adjoint.
    EXPECT_NEAR(sx.dense_matrix_f64[0].real(), 0.5, 1e-15);
    EXPECT_NEAR(sx.dense_matrix_f64[0].imag(), 0.5, 1e-15);
    EXPECT_NEAR(sx.dense_matrix_f64[1].imag(), -0.5, 1e-15);
    for (int k = 0; k < 4; ++k) {
        int t = (k % 2) * 2 + k / 2;
        EXPECT_NEAR(std::abs(sxdg.dense_matrix_f64[k] - std::conj(sx.dense_matrix_f64[t])), 0.0, 1e-15);
    }
    // SX · SX = X on |0⟩ → |1⟩.
    auto twice = decoder.Decode(R"({
      "qsal_version": "0.1.0",
      "circuit": {"num_qubits": 1, "num_clbits": 0, "instructions": [
        {"type": "GATE", "qubits": [0], "gate": {"name": "sx", "gate_type": "standard"}},
        {"type": "GATE", "qubits": [0], "gate": {"name": "sx", "gate_type": "standard"}}
      ]}
    })");
    ASSERT_NE(twice, nullptr) << decoder.LastError();
    EXPECT_TRUE(ir::Validator::Validate(*twice).empty());
}

TEST(QSALDecoderTest, DecodeSymbolicAndAffineParameters) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [
          {"type": "GATE", "qubits": [0], "gate": {
            "name": "rx", "gate_type": "parameterized",
            "params": [{"_param": "theta"}]
          }},
          {"type": "GATE", "qubits": [0], "gate": {
            "name": "ry", "gate_type": "parameterized",
            "params": [{"_expr": {
              "constant": 0.25,
              "terms": {"beta": -2.0, "theta": 0.5, "ignored": 0.0}
            }}]
          }},
          {"type": "GATE", "qubits": [0], "gate": {
            "name": "rz", "gate_type": "parameterized", "params": [0.125]
          }}
        ]
      }
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->ops.size(), 3);
    ASSERT_EQ(prog->parameter_names.size(), 2);
    EXPECT_EQ(prog->parameter_names[0], "theta");
    EXPECT_EQ(prog->parameter_names[1], "beta");

    const auto& rx = std::get<runtime::GateOperation>(prog->ops[0]);
    EXPECT_TRUE(rx.params_f64.empty());
    ASSERT_EQ(rx.parameter_expressions.size(), 1);
    EXPECT_DOUBLE_EQ(rx.parameter_expressions[0].constant, 0.0);
    ASSERT_EQ(rx.parameter_expressions[0].terms.size(), 1);
    EXPECT_EQ(rx.parameter_expressions[0].terms[0].first, "theta");
    EXPECT_DOUBLE_EQ(rx.parameter_expressions[0].terms[0].second, 1.0);

    const auto& ry = std::get<runtime::GateOperation>(prog->ops[1]);
    EXPECT_TRUE(ry.params_f64.empty());
    ASSERT_EQ(ry.parameter_expressions.size(), 1);
    EXPECT_DOUBLE_EQ(ry.parameter_expressions[0].constant, 0.25);
    ASSERT_EQ(ry.parameter_expressions[0].terms.size(), 2);
    EXPECT_EQ(ry.parameter_expressions[0].terms[0].first, "beta");
    EXPECT_DOUBLE_EQ(ry.parameter_expressions[0].terms[0].second, -2.0);
    EXPECT_EQ(ry.parameter_expressions[0].terms[1].first, "theta");
    EXPECT_DOUBLE_EQ(ry.parameter_expressions[0].terms[1].second, 0.5);

    const auto& rz = std::get<runtime::GateOperation>(prog->ops[2]);
    ASSERT_EQ(rz.params_f64.size(), 1);
    EXPECT_DOUBLE_EQ(rz.params_f64[0], 0.125);
    ASSERT_EQ(rz.parameter_expressions.size(), 1);
    EXPECT_DOUBLE_EQ(rz.parameter_expressions[0].constant, 0.125);
    EXPECT_FALSE(rz.parameter_expressions[0].IsSymbolic());
}

TEST(QSALDecoderTest, RejectMalformedAffineParameter) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [{"type": "GATE", "qubits": [0], "gate": {
          "name": "rx", "gate_type": "parameterized",
          "params": [{"_expr": {"constant": 0.0, "terms": {"theta": "bad"}}}]
        }}]
      }
    })";

    ir::QSALDecoder decoder;
    EXPECT_EQ(decoder.Decode(source), nullptr);
    EXPECT_NE(decoder.LastError().find("numeric coefficients"), std::string::npos);
}

TEST(QSALDecoderTest, DecodeWithObservables) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kWithObservable);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->observables.size(), 2);
    EXPECT_EQ(prog->observables[0].label, "ZZ");
    EXPECT_DOUBLE_EQ(std::real(prog->observables[0].coefficient), 1.0);
    ASSERT_EQ(prog->observables[0].paulis.size(), 2);

    EXPECT_EQ(prog->observables[1].label, "XX");
}

TEST(QSALDecoderTest, DecodeMultiTermObservablePreservesEveryTerm) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {"num_qubits": 1, "num_clbits": 0, "instructions": []},
      "observables": [{
        "name": "energy",
        "operator": {"terms": [
          {"paulis": [[0, "Z"]], "coefficient": 0.5},
          {"paulis": [[0, "X"]], "coefficient": -0.25},
          {"paulis": [[0, "Y"]], "coefficient": 0.125}
        ]}
      }]
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->observables.size(), 3);
    for (const auto& term : prog->observables) {
        EXPECT_EQ(term.label, "energy");
        ASSERT_EQ(term.paulis.size(), 1);
    }
    EXPECT_EQ(prog->observables[0].paulis[0].second, 'Z');
    EXPECT_DOUBLE_EQ(std::real(prog->observables[0].coefficient), 0.5);
    EXPECT_EQ(prog->observables[1].paulis[0].second, 'X');
    EXPECT_DOUBLE_EQ(std::real(prog->observables[1].coefficient), -0.25);
    EXPECT_EQ(prog->observables[2].paulis[0].second, 'Y');
    EXPECT_DOUBLE_EQ(std::real(prog->observables[2].coefficient), 0.125);
}

TEST(QSALDecoderTest, DecodeComplexObservableCoefficientPair) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {"num_qubits": 2, "num_clbits": 0, "instructions": []},
      "observables": [{
        "name": "H",
        "operator": {"terms": [
          {"paulis": [[0, "Z"]], "coefficient": [0.5, 0.5]},
          {"paulis": [[1, "X"]], "coefficient": [0.0, 1.0]},
          {"paulis": [[1, "Z"]], "coefficient": 2.0}
        ]}
      }]
    })";
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->observables.size(), 3);
    EXPECT_EQ(prog->observables[0].coefficient, std::complex<double>(0.5, 0.5));
    EXPECT_EQ(prog->observables[1].coefficient, std::complex<double>(0.0, 1.0));
    EXPECT_EQ(prog->observables[2].coefficient, std::complex<double>(2.0, 0.0));

    executor::SimulationExecutor exec({});
    executor::ExecutionRequest req;
    req.return_expectations = true;
    const auto result = exec.Run(*prog, req);
    // |00>: <Z0> = 1, <X1> = 0, <Z1> = 1 -> (0.5 + 0.5i) + 0 + 2
    ASSERT_EQ(result.expectations_complex.count("H"), 1);
    EXPECT_NEAR(std::real(result.expectations_complex.at("H")), 2.5, 1e-12);
    EXPECT_NEAR(std::imag(result.expectations_complex.at("H")), 0.5, 1e-12);
    EXPECT_NEAR(result.expectations_real.at("H"), 2.5, 1e-12);
}

TEST(QSALDecoderTest, RejectMalformedComplexObservableCoefficient) {
    for (const std::string coefficient : {"[1.0]", "[\"a\", 1.0]", "[1.0, 2.0, 3.0]", "\"t\""}) {
        const std::string source = R"({
          "qsal_version": "0.1.0",
          "circuit": {"num_qubits": 1, "num_clbits": 0, "instructions": []},
          "observables": [{"name": "H", "operator": {"terms": [
            {"paulis": [[0, "Z"]], "coefficient": )"
                                   + coefficient + R"(}]}}]
        })";
        ir::QSALDecoder decoder;
        EXPECT_EQ(decoder.Decode(source), nullptr) << coefficient;
        EXPECT_NE(decoder.LastError().find("non-numeric coefficient"), std::string::npos) << coefficient;
    }
}

TEST(QSALDecoderTest, IgnoresOptionalCircuitNameAndGateLabel) {
    // Python QSAL 会写出可选的 circuit.name 与 gate.label，native 解码器应忽略它们。
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "name": "document",
      "circuit": {
        "name": "ansatz",
        "num_qubits": 1,
        "num_clbits": 0,
        "instructions": [
          {"type": "GATE", "qubits": [0], "clbits": [],
           "gate": {"name": "h", "num_qubits": 1, "gate_type": "standard", "label": "hadamard"}},
          {"type": "GATE", "qubits": [0], "clbits": [],
           "gate": {"name": "rx", "num_qubits": 1, "params": [0.3], "gate_type": "parameterized",
                    "label": "drive"}}
        ]
      }
    })";
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();
    EXPECT_EQ(prog->num_qubits, 1);
    EXPECT_EQ(prog->ops.size(), 2);
}

TEST(QSALDecoderTest, DecodeBarrierCircuit) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kBarrierCircuit);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->ops.size(), 3);

    // Barrier 应该被解析为 kBarrier
    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[1]));
    const auto& barrier = std::get<runtime::GateOperation>(prog->ops[1]);
    EXPECT_EQ(barrier.opcode, runtime::OpCode::kBarrier);
    ASSERT_EQ(barrier.qubits.size(), 2);
    EXPECT_EQ(barrier.qubits[0], 0);
    EXPECT_EQ(barrier.qubits[1], 1);
}

TEST(QSALDecoderTest, DecodeChannelPreservesCanonicalKindParametersAndKrausData) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [{
          "type": "CHANNEL", "qubits": [0], "clbits": [],
          "channel": {
            "channel_type": "custom_y",
            "channel_kind": "pauli",
            "num_qubits": 1,
            "px": 0.0, "py": 1.0, "pz": 0.0,
            "kraus_ops": [[
              [[0.0, 0.0], [0.0, -1.0]],
              [[0.0, 1.0], [0.0, 0.0]]
            ]]
          }
        }]
      }
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    ASSERT_EQ(prog->ops.size(), 1);
    ASSERT_TRUE(std::holds_alternative<runtime::ChannelOperation>(prog->ops[0]));
    const auto& channel = std::get<runtime::ChannelOperation>(prog->ops[0]);
    EXPECT_EQ(channel.channel_kind, "pauli");
    EXPECT_EQ(channel.qubits, std::vector<uint32_t>({0}));
    ASSERT_EQ(channel.params_f64.size(), 3);
    EXPECT_DOUBLE_EQ(channel.params_f64[0], 0.0);
    EXPECT_DOUBLE_EQ(channel.params_f64[1], 1.0);
    EXPECT_DOUBLE_EQ(channel.params_f64[2], 0.0);
    EXPECT_EQ(channel.num_kraus_ops, 1);
    ASSERT_EQ(channel.kraus_operators_f64.size(), 4);
    EXPECT_EQ(channel.kraus_operators_f64[0], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(channel.kraus_operators_f64[1], std::complex<double>(0.0, -1.0));
    EXPECT_EQ(channel.kraus_operators_f64[2], std::complex<double>(0.0, 1.0));
    EXPECT_EQ(channel.kraus_operators_f64[3], std::complex<double>(0.0, 0.0));
    ASSERT_EQ(channel.kraus_operators_f32.size(), 4);
    EXPECT_EQ(channel.kraus_operators_f32[1], std::complex<float>(0.0F, -1.0F));
    EXPECT_TRUE(channel.use_pauli_fast_path);
}

TEST(QSALDecoderTest, ConflictingChannelMetadataFallsBackToSerializedKrausMap) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [{
          "type": "CHANNEL", "qubits": [0],
          "channel": {
            "channel_kind": "bit_flip", "num_qubits": 1, "p": 0.0,
            "kraus_ops": [[
              [[0.0, 0.0], [1.0, 0.0]],
              [[1.0, 0.0], [0.0, 0.0]]
            ]]
          }
        }]
      }
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);

    ASSERT_NE(prog, nullptr) << decoder.LastError();
    const auto& channel = std::get<runtime::ChannelOperation>(prog->ops[0]);
    EXPECT_FALSE(channel.use_pauli_fast_path);
    EXPECT_TRUE(ir::Validator::Validate(*prog).empty());
}

TEST(QSALDecoderTest, RejectMalformedChannelKrausShape) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [{
          "type": "CHANNEL", "qubits": [0],
          "channel": {
            "channel_kind": "kraus", "num_qubits": 1,
            "kraus_ops": [[[1.0, 0.0]]]
          }
        }]
      }
    })";

    ir::QSALDecoder decoder;
    EXPECT_EQ(decoder.Decode(source), nullptr);
    EXPECT_NE(decoder.LastError().find("wrong row count"), std::string::npos);
}

TEST(QSALDecoderTest, InvalidJSON) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode("not json at all");
    EXPECT_EQ(prog, nullptr);
    EXPECT_FALSE(decoder.LastError().empty());
}

TEST(QSALDecoderTest, UnsupportedVersion) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(R"({"qsal_version": "9.0.0", "circuit": null})");
    EXPECT_EQ(prog, nullptr);
    EXPECT_TRUE(decoder.LastError().find("unsupported") != std::string::npos);
}

TEST(QSALDecoderTest, UnknownGate) {
    ir::QSALDecoder decoder;
    std::string json = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [
          {"type": "GATE", "qubits": [0], "clbits": [],
           "gate": {"name": "unknown_gate", "num_qubits": 1, "gate_type": "standard"}}
        ]
      }
    })";
    auto prog = decoder.Decode(json);
    EXPECT_EQ(prog, nullptr);
    EXPECT_TRUE(decoder.LastError().find("unknown") != std::string::npos);
}

TEST(QSALDecoderTest, MissingCircuit) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(R"({"qsal_version": "0.1.0"})");
    EXPECT_EQ(prog, nullptr);
    EXPECT_TRUE(decoder.LastError().find("missing") != std::string::npos);
}

// ============================================================
// Validator 测试
// ============================================================

TEST(ValidatorTest, EmptyProgramIsValid) {
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 1;

    auto errors = ir::Validator::Validate(prog);
    EXPECT_TRUE(errors.empty());
}

TEST(ValidatorTest, QubitIndexOutOfBounds) {
    auto prog = ir::ProgramBuilder(2, 1)
                    .AddGate("h", {3})  // qubit 3 >= num_qubits 2
                    .Build();

    auto errors = ir::Validator::CheckIndexBounds(*prog);
    ASSERT_EQ(errors.size(), 1);
    EXPECT_TRUE(errors[0].find(">= num_qubits") != std::string::npos);
}

TEST(ValidatorTest, ClbitIndexOutOfBounds) {
    auto prog = ir::ProgramBuilder(2, 1)
                    .AddMeasure({0}, {5})  // clbit 5 >= num_clbits 1
                    .Build();

    auto errors = ir::Validator::CheckIndexBounds(*prog);
    ASSERT_EQ(errors.size(), 1);
    EXPECT_TRUE(errors[0].find(">= num_clbits") != std::string::npos);
}

TEST(ValidatorTest, UnknownGateName) {
    auto prog = ir::ProgramBuilder(2, 0).AddGate("unknown_gate", {0}).Build();

    auto errors = ir::Validator::CheckGateNames(*prog);
    ASSERT_EQ(errors.size(), 1);
    EXPECT_TRUE(errors[0].find("unknown gate") != std::string::npos);
}

TEST(ValidatorTest, KnownGateOK) {
    auto prog = ir::ProgramBuilder(2, 0).AddGate("h", {0}).AddGate("cx", {0, 1}).Build();

    auto errors = ir::Validator::CheckGateNames(*prog);
    EXPECT_TRUE(errors.empty());
}

TEST(ValidatorTest, ParameterCountMismatch) {
    auto prog = ir::ProgramBuilder(1, 0)
                    .AddGate("rx", {0}, {})  // rx needs 1 param, got 0
                    .Build();

    auto errors = ir::Validator::CheckParameterCount(*prog);
    ASSERT_EQ(errors.size(), 1);
    EXPECT_TRUE(errors[0].find("expected") != std::string::npos);
}

TEST(ValidatorTest, SymbolicParameterExpressionSatisfiesParameterCount) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    runtime::GateOperation rx;
    rx.opcode = runtime::OpCode::kNamedGate;
    rx.name = "rx";
    rx.qubits = {0};
    runtime::ParameterExpression expression;
    expression.terms.emplace_back("theta", 1.0);
    rx.parameter_expressions.push_back(expression);
    prog.ops.emplace_back(std::move(rx));

    EXPECT_TRUE(ir::Validator::CheckParameterCount(prog).empty());
}

TEST(ValidatorTest, QubitCountMismatch) {
    auto prog = ir::ProgramBuilder(3, 0)
                    .AddGate("cx", {0})  // cx needs 2 qubits, got 1
                    .Build();

    auto errors = ir::Validator::CheckParameterCount(*prog);
    ASSERT_EQ(errors.size(), 1);
    EXPECT_TRUE(errors[0].find("expected") != std::string::npos);
}

TEST(ValidatorTest, FullValidationAggregatesErrors) {
    auto prog = ir::ProgramBuilder(1, 0).AddGate("bad_gate", {5}).AddMeasure({0}, {10}).Build();

    auto errors = ir::Validator::Validate(*prog);
    EXPECT_GE(errors.size(), 3);
}

TEST(ValidatorTest, RejectNonTracePreservingKrausChannel) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [{
          "type": "CHANNEL", "qubits": [0],
          "channel": {
            "channel_kind": "kraus", "num_qubits": 1,
            "kraus_ops": [[
              [[0.5, 0.0], [0.0, 0.0]],
              [[0.0, 0.0], [0.5, 0.0]]
            ]]
          }
        }]
      }
    })";

    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();

    const auto errors = ir::Validator::Validate(*prog);
    ASSERT_EQ(errors.size(), 1);
    EXPECT_NE(errors[0].find("not trace-preserving"), std::string::npos);
}

// ============================================================
// ProgramBuilder 测试
// ============================================================

TEST(ProgramBuilderTest, BuilderCreatesValidProgram) {
    auto prog = ir::ProgramBuilder(3, 2)
                    .AddGate("h", {0})
                    .AddGate("cx", {0, 1})
                    .AddGate("rx", {2}, {0.5})
                    .AddBarrier({0, 1, 2})
                    .AddMeasure({0, 1}, {0, 1})
                    .AddObservable("ZZ", {{0, 'Z'}, {1, 'Z'}})
                    .Build();

    EXPECT_EQ(prog->num_qubits, 3);
    EXPECT_EQ(prog->num_clbits, 2);
    EXPECT_EQ(prog->ops.size(), 5);
    EXPECT_EQ(prog->observables.size(), 1);

    // 验证 Gate 名称
    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[0]));
    EXPECT_EQ(std::get<runtime::GateOperation>(prog->ops[0]).name, "h");

    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[1]));
    EXPECT_EQ(std::get<runtime::GateOperation>(prog->ops[1]).name, "cx");

    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[2]));
    const auto& rx = std::get<runtime::GateOperation>(prog->ops[2]);
    EXPECT_EQ(rx.name, "rx");
    EXPECT_DOUBLE_EQ(rx.params_f64[0], 0.5);
    ASSERT_EQ(rx.parameter_expressions.size(), 1);
    EXPECT_DOUBLE_EQ(rx.parameter_expressions[0].constant, 0.5);
    EXPECT_FALSE(rx.parameter_expressions[0].IsSymbolic());

    ASSERT_TRUE(std::holds_alternative<runtime::GateOperation>(prog->ops[3]));
    EXPECT_EQ(std::get<runtime::GateOperation>(prog->ops[3]).opcode, runtime::OpCode::kBarrier);

    ASSERT_TRUE(std::holds_alternative<runtime::MeasureOperation>(prog->ops[4]));
}

TEST(ProgramBuilderTest, ObservableCoefficient) {
    auto prog = ir::ProgramBuilder(1, 0).AddObservable("Z0", {{0, 'Z'}}, std::complex<double>(2.5, 0)).Build();

    ASSERT_EQ(prog->observables.size(), 1);
    EXPECT_EQ(prog->observables[0].label, "Z0");
    EXPECT_DOUBLE_EQ(std::real(prog->observables[0].coefficient), 2.5);
    ASSERT_EQ(prog->observables[0].paulis.size(), 1);
    EXPECT_EQ(prog->observables[0].paulis[0].first, 0);
    EXPECT_EQ(prog->observables[0].paulis[0].second, 'Z');
}

// ============================================================
// 端到端测试: QSAL JSON → decode → execute
// ============================================================

TEST(EndToEndTest, HHGateCreatesPlusState) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kHHGate);
    ASSERT_NE(prog, nullptr);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = exec.Run(*prog, req);

    // |+⟩ = (|0⟩ + |1⟩)/√2
    ASSERT_EQ(result.statevector_f64.size(), 2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), inv_sqrt2, 1e-10);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), inv_sqrt2, 1e-10);
}

TEST(EndToEndTest, RXPiIsXGate) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kRXGate);
    ASSERT_NE(prog, nullptr);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = exec.Run(*prog, req);

    // RX(π)|0⟩ ≈ -i|1⟩ (up to global phase)
    ASSERT_EQ(result.statevector_f64.size(), 2);
    EXPECT_NEAR(std::abs(result.statevector_f64[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.statevector_f64[1]), 1.0, 1e-10);
}

TEST(EndToEndTest, BellStateHasCorrectCounts) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kBellState);
    ASSERT_NE(prog, nullptr);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 200;
    req.return_samples = true;

    auto result = exec.Run(*prog, req);

    ASSERT_EQ(result.packed_samples.size(), 200);

    // Bell 态测量应只产生 00 或 11
    for (auto s : result.packed_samples) {
        EXPECT_TRUE(s == 0 || s == 3) << "Unexpected sample: " << s;
    }
}

TEST(EndToEndTest, ObservableExpectation) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kWithObservable);
    ASSERT_NE(prog, nullptr);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(*prog, req);

    // Bell 态: ⟨Z₀Z₁⟩ = 1, ⟨X₀X₁⟩ = 1
    ASSERT_TRUE(result.expectations_real.count("ZZ"));
    ASSERT_TRUE(result.expectations_real.count("XX"));
    EXPECT_NEAR(result.expectations_real.at("ZZ"), 1.0, 1e-10);
    EXPECT_NEAR(result.expectations_real.at("XX"), 1.0, 1e-10);
}

TEST(EndToEndTest, MultiTermHamiltonianIsAggregatedByLabel) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {"num_qubits": 1, "num_clbits": 0, "instructions": []},
      "observables": [{
        "name": "energy",
        "operator": {"terms": [
          {"paulis": [[0, "Z"]], "coefficient": 0.5},
          {"paulis": [[0, "X"]], "coefficient": 0.25}
        ]}
      }]
    })";
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();

    executor::SimulationExecutor exec({});
    executor::ExecutionRequest req;
    req.return_expectations = true;
    const auto result = exec.Run(*prog, req);

    ASSERT_EQ(result.expectations_real.count("energy"), 1);
    EXPECT_NEAR(result.expectations_real.at("energy"), 0.5, 1e-10);
    ASSERT_EQ(result.expectations_complex.count("energy"), 1);
    EXPECT_NEAR(std::real(result.expectations_complex.at("energy")), 0.5, 1e-10);
}

TEST(EndToEndTest, SimulationExecutorRejectsUnboundSymbolicProgram) {
    const std::string source = R"({
      "qsal_version": "0.1.0",
      "circuit": {
        "num_qubits": 1, "num_clbits": 0,
        "instructions": [{"type": "GATE", "qubits": [0], "gate": {
          "name": "rx", "gate_type": "parameterized", "params": [{"_param": "theta"}]
        }}]
      }
    })";
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(source);
    ASSERT_NE(prog, nullptr) << decoder.LastError();

    executor::SimulationExecutor exec({});
    executor::ExecutionRequest req;
    EXPECT_THROW(exec.Run(*prog, req), std::invalid_argument);
}

TEST(EndToEndTest, BarrierDoesNotAffectExecution) {
    ir::QSALDecoder decoder;
    auto prog = decoder.Decode(kBarrierCircuit);
    ASSERT_NE(prog, nullptr);

    // Barrier 不应影响线路执行结果
    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = exec.Run(*prog, req);

    // H + CX 产生 Bell 态，statevector 应归一化
    double norm = 0;
    for (auto& v : result.statevector_f64)
        norm += std::norm(v);
    EXPECT_NEAR(norm, 1.0, 1e-10);
}
