#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "quspect_native/ir/circuit.hpp"

namespace quspect::native::ir {

// OpenQASM 2.0 解析器（quspect.ir.qasm_importer 的 C++ 快速路径）。
//
// 语义与 Python 参考实现逐项一致：多个 qreg/creg 按声明顺序拼接、整寄存器广播、完整的参数
// 表达式（+ - * / ^、一元正负号、pi、sin cos tan exp ln sqrt）、内置 U/CX 与 qelib1.inc 门库
// （有 QuSpect 对应门的直接导入，其余按库定义展开）、带形参的用户 gate 声明（递归展开）。
//
// 失败约定：凡是参考实现会报错的输入，本解析器一定返回 false；另外对少数边角情况（非 ASCII
// 字符、超出 64 位的整数、幂 / 函数的非有限输入或结果等）也保守地返回 false。调用方遇到 false
// 时改用参考实现解析，从而得到与原来完全相同的结果或异常（含行列号）。

struct Qasm2Options {
    // 实数字面量的解析函数，必须与宿主语言的 float() 逐位一致；为空时用 "C" locale 的
    // std::strtod（正确舍入）。
    double (*parse_real)(const char* text, std::size_t length) = nullptr;
};

// 编译好的内置门库（U、CX、qelib1.inc 中有 QuSpect 对应门的门，以及 definitions 中以
// OpenQASM gate 声明给出的其余库门）。
class Qasm2Library;

// 编译门库；definitions 是一串 gate 声明（与 Python 端 _QELIB1_DEFINITIONS 相同）。
// 抛出 std::invalid_argument：definitions 无法解析。
std::shared_ptr<const Qasm2Library> CompileQasm2Library(std::string_view definitions,
                                                        const Qasm2Options& options = {});

// 解析一段 OpenQASM 2.0 程序，指令追加到 out（out 应为空）。成功返回 true；否则返回 false，
// error 中给出原因（只用于调试，不保证与参考实现的报错文字相同）。
bool ParseQasm2(std::string_view source, const Qasm2Library& library, Circuit& out, std::string* error = nullptr,
                const Qasm2Options& options = {});

}  // namespace quspect::native::ir
