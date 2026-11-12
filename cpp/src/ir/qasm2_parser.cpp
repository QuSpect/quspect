#include "quspect_native/ir/qasm2_parser.hpp"

#include <cmath>

#include "quspect_native/detmath/scalar.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// 本文件中的浮点运算必须与 Python 参考实现逐位一致：CMake 为它关闭了浮点收缩（FMA 融合）。

namespace quspect::native::ir {

namespace {

// ---------------------------------------------------------------------------
// 词法
// ---------------------------------------------------------------------------

enum class Tok : std::uint8_t { kReal, kInt, kId, kString, kOp, kEof };

struct Token {
    Tok kind;
    std::uint32_t start;
    std::uint32_t length;
};

// 解析失败（参考实现会报错，或本解析器不处理）。
struct Reject {
    std::string reason;
};

[[noreturn]] void Fail(std::string reason) {
    throw Reject{std::move(reason)};
}

bool IsDigit(char c) {
    return c >= '0' && c <= '9';
}
bool IsIdStart(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}
bool IsIdChar(char c) {
    return IsIdStart(c) || IsDigit(c);
}

// 与参考实现的正则表达式（按候选顺序匹配）一致的分词。
std::vector<Token> Tokenize(std::string_view src) {
    std::vector<Token> tokens;
    tokens.reserve(src.size() / 3 + 8);
    const std::size_t n = src.size();
    std::size_t pos = 0;
    auto push = [&](Tok kind, std::size_t start, std::size_t end) {
        tokens.push_back(Token{kind, static_cast<std::uint32_t>(start), static_cast<std::uint32_t>(end - start)});
    };
    // 匹配 [eE][+-]?\d+，返回结束位置（不匹配时返回 p）。
    auto exponent = [&](std::size_t p) {
        if (p < n && (src[p] == 'e' || src[p] == 'E')) {
            std::size_t q = p + 1;
            if (q < n && (src[q] == '+' || src[q] == '-')) {
                ++q;
            }
            if (q < n && IsDigit(src[q])) {
                while (q < n && IsDigit(src[q])) {
                    ++q;
                }
                return q;
            }
        }
        return p;
    };
    while (pos < n) {
        const char c = src[pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' || c == '\n') {
            ++pos;
            continue;
        }
        if (c == '/' && pos + 1 < n && src[pos + 1] == '/') {
            while (pos < n && src[pos] != '\n') {
                ++pos;
            }
            continue;
        }
        if (c == '"') {
            std::size_t q = pos + 1;
            while (q < n && src[q] != '"' && src[q] != '\n') {
                ++q;
            }
            if (q >= n || src[q] != '"') {
                Fail("invalid character '\"'");
            }
            push(Tok::kString, pos, q + 1);
            pos = q + 1;
            continue;
        }
        if (IsDigit(c) || (c == '.' && pos + 1 < n && IsDigit(src[pos + 1]))) {
            // real: (\d+\.\d*|\.\d+)([eE][+-]?\d+)? | \d+[eE][+-]?\d+ ；否则 int: \d+
            std::size_t q = pos;
            while (q < n && IsDigit(src[q])) {
                ++q;
            }
            if (q < n && src[q] == '.') {
                std::size_t r = q + 1;
                while (r < n && IsDigit(src[r])) {
                    ++r;
                }
                r = exponent(r);
                push(Tok::kReal, pos, r);
                pos = r;
                continue;
            }
            const std::size_t e = exponent(q);
            if (e != q) {
                push(Tok::kReal, pos, e);
                pos = e;
                continue;
            }
            push(Tok::kInt, pos, q);
            pos = q;
            continue;
        }
        if (IsIdStart(c)) {
            std::size_t q = pos + 1;
            while (q < n && IsIdChar(src[q])) {
                ++q;
            }
            push(Tok::kId, pos, q);
            pos = q;
            continue;
        }
        if (c == '-' && pos + 1 < n && src[pos + 1] == '>') {
            push(Tok::kOp, pos, pos + 2);
            pos += 2;
            continue;
        }
        if (c == '=' && pos + 1 < n && src[pos + 1] == '=') {
            push(Tok::kOp, pos, pos + 2);
            pos += 2;
            continue;
        }
        if (std::strchr("[](){},;+-*/^", c) != nullptr && c != '\0') {
            push(Tok::kOp, pos, pos + 1);
            ++pos;
            continue;
        }
        Fail("invalid character");
    }
    push(Tok::kEof, n, n);
    return tokens;
}

// ---------------------------------------------------------------------------
// 参数表达式
// ---------------------------------------------------------------------------

enum class ExprOp : std::uint8_t { kConst, kParam, kNeg, kAdd, kSub, kMul, kDiv, kPow, kFunc };
enum class Func : std::uint8_t { kSin, kCos, kTan, kExp, kLn, kSqrt };

struct ExprNode {
    ExprOp op;
    Func func = Func::kSin;
    double value = 0.0;
    std::int32_t a = -1;
    std::int32_t b = -1;
};

// 表达式存放在所属门定义（或顶层语句）的节点池中，以下标引用。
using ExprPool = std::vector<ExprNode>;

bool IsOddInteger(double w) {
    // 与 CPython 的 DOUBLE_IS_ODD_INTEGER 相同：w 为整数且 fmod(|w|, 2) == 1。
    return std::fmod(std::fabs(w), 2.0) == 1.0;
}

// CPython float.__pow__ 对有限输入的语义（float_pow）；非有限输入、复数结果、除零与溢出一律失败。
double PythonPow(double iv, double iw) {
    if (!std::isfinite(iv) || !std::isfinite(iw)) {
        Fail("non-finite power operand");
    }
    if (iw == 0.0) {
        return 1.0;
    }
    if (iv == 0.0) {
        if (iw < 0.0) {
            Fail("zero to a negative power");
        }
        return IsOddInteger(iw) ? iv : 0.0;
    }
    bool negate = false;
    if (iv < 0.0) {
        if (iw != std::floor(iw)) {
            Fail("complex power");
        }
        iv = -iv;
        negate = IsOddInteger(iw);
    }
    if (iv == 1.0) {
        return negate ? -1.0 : 1.0;
    }
    double ix = detmath::Pow(iv, iw);
    if (std::isinf(ix)) {
        Fail("power overflow");
    }
    return negate ? -ix : ix;
}

double ApplyFunc(Func f, double x) {
    if (!std::isfinite(x)) {
        Fail("non-finite function argument");
    }
    double r = 0.0;
    switch (f) {
        case Func::kSin:
            r = detmath::Sin(x);
            break;
        case Func::kCos:
            r = detmath::Cos(x);
            break;
        case Func::kTan:
            r = detmath::Tan(x);
            break;
        case Func::kExp:
            r = detmath::Exp(x);
            break;
        case Func::kLn:
            if (!(x > 0.0)) {
                Fail("log domain");
            }
            r = detmath::Log(x);
            break;
        case Func::kSqrt:
            if (x < 0.0) {
                Fail("sqrt domain");
            }
            r = std::sqrt(x);
            break;
    }
    if (!std::isfinite(r)) {
        Fail("non-finite function result");
    }
    return r;
}

double Eval(const ExprPool& pool, std::int32_t idx, const double* env) {
    const ExprNode& e = pool[static_cast<std::size_t>(idx)];
    switch (e.op) {
        case ExprOp::kConst:
            return e.value;
        case ExprOp::kParam:
            return env[e.a];
        case ExprOp::kNeg:
            return -Eval(pool, e.a, env);
        case ExprOp::kAdd: {
            const double l = Eval(pool, e.a, env);
            const double r = Eval(pool, e.b, env);
            return l + r;
        }
        case ExprOp::kSub: {
            const double l = Eval(pool, e.a, env);
            const double r = Eval(pool, e.b, env);
            return l - r;
        }
        case ExprOp::kMul: {
            const double l = Eval(pool, e.a, env);
            const double r = Eval(pool, e.b, env);
            return l * r;
        }
        case ExprOp::kDiv: {
            const double r = Eval(pool, e.b, env);
            if (r == 0.0) {
                Fail("division by zero");
            }
            const double l = Eval(pool, e.a, env);
            return l / r;
        }
        case ExprOp::kPow: {
            const double l = Eval(pool, e.a, env);
            const double r = Eval(pool, e.b, env);
            return PythonPow(l, r);
        }
        case ExprOp::kFunc:
            return ApplyFunc(e.func, Eval(pool, e.a, env));
    }
    Fail("bad expression");
}

// ---------------------------------------------------------------------------
// 门定义
// ---------------------------------------------------------------------------

struct GateDef;

struct BodyOp {
    std::shared_ptr<const GateDef> gate;
    std::vector<std::int32_t> params;  // 表达式根节点（在所属 GateDef 的 pool 中）
    std::vector<std::uint32_t> qubits;  // 形参比特下标
};

struct GateDef {
    std::string name;
    std::vector<std::string> params;
    std::uint32_t num_qubits = 0;
    std::vector<BodyOp> body;
    ExprPool pool;
    bool opaque = false;
    bool builtin = false;
    std::optional<OpKind> native;  // builtin 且有 QuSpect 对应门
};

using GateTable = std::unordered_map<std::string, std::shared_ptr<const GateDef>>;

struct NativeGate {
    const char* qasm_name;
    OpKind kind;
    std::uint32_t num_params;
    std::uint32_t num_qubits;
};

// QASM 门名 -> QuSpect 门（与 Python 端 _NATIVE_GATES 相同，矩阵完全一致）。
constexpr NativeGate kNativeGates[] = {
    {"U", OpKind::kU3, 3, 1},     {"CX", OpKind::kCX, 0, 2},     {"id", OpKind::kId, 0, 1},
    {"h", OpKind::kH, 0, 1},      {"x", OpKind::kX, 0, 1},       {"y", OpKind::kY, 0, 1},
    {"z", OpKind::kZ, 0, 1},      {"s", OpKind::kS, 0, 1},       {"sdg", OpKind::kSdg, 0, 1},
    {"t", OpKind::kT, 0, 1},      {"tdg", OpKind::kTdg, 0, 1},   {"sx", OpKind::kSX, 0, 1},
    {"sxdg", OpKind::kSXdg, 0, 1}, {"cx", OpKind::kCX, 0, 2},    {"cz", OpKind::kCZ, 0, 2},
    {"swap", OpKind::kSwap, 0, 2}, {"ccx", OpKind::kCCX, 0, 3},  {"rx", OpKind::kRX, 1, 1},
    {"ry", OpKind::kRY, 1, 1},    {"rz", OpKind::kRZ, 1, 1},     {"p", OpKind::kP, 1, 1},
    {"u1", OpKind::kP, 1, 1},     {"u2", OpKind::kU2, 2, 1},     {"u3", OpKind::kU3, 3, 1},
    {"u", OpKind::kU3, 3, 1},     {"crx", OpKind::kCRX, 1, 2},   {"cry", OpKind::kCRY, 1, 2},
    {"crz", OpKind::kCRZ, 1, 2},  {"cp", OpKind::kCP, 1, 2},     {"cu1", OpKind::kCP, 1, 2},
    {"cu3", OpKind::kCU3, 3, 2},  {"rxx", OpKind::kRXX, 1, 2},   {"rzz", OpKind::kRZZ, 1, 2},
};

struct Register {
    std::uint32_t offset;
    std::uint32_t size;
};

// 操作数：全局比特列表 + 是否整寄存器。
struct Operand {
    std::vector<std::uint32_t> bits;
    bool whole;
};

constexpr std::uint64_t kMaxBits = std::uint64_t{1} << 31;

class Parser {
public:
    Parser(std::string_view src, GateTable gates, const Qasm2Options& options, Circuit* out)
        : src_(src), tokens_(Tokenize(src)), gates_(std::move(gates)), options_(options), out_(out) {
    }

    void parse_program() {
        expect_keyword("OPENQASM");
        const Token& version = peek();
        if (version.kind != Tok::kReal && version.kind != Tok::kInt) {
            Fail("bad version");
        }
        if (parse_real(version) != 2.0) {
            Fail("bad version");
        }
        advance();
        expect_op(";");
        while (peek().kind != Tok::kEof) {
            parse_statement();
        }
    }

    void parse_library() {
        while (peek().kind != Tok::kEof) {
            parse_gate_declaration(true);
        }
    }

    GateTable take_gates() {
        return std::move(gates_);
    }

private:
    // ---- 词法辅助 ----
    const Token& peek(std::size_t offset = 0) const {
        const std::size_t idx = std::min(pos_ + offset, tokens_.size() - 1);
        return tokens_[idx];
    }
    const Token& advance() {
        const Token& tok = tokens_[pos_];
        if (tok.kind != Tok::kEof) {
            ++pos_;
        }
        return tok;
    }
    std::string_view text(const Token& tok) const {
        return src_.substr(tok.start, tok.length);
    }
    bool is_op(std::string_view value, std::size_t offset = 0) const {
        const Token& tok = peek(offset);
        return tok.kind == Tok::kOp && text(tok) == value;
    }
    bool is_id(const Token& tok, std::string_view value) const {
        return tok.kind == Tok::kId && text(tok) == value;
    }
    const Token& expect_op(std::string_view value) {
        if (!is_op(value)) {
            Fail("expected operator");
        }
        return advance();
    }
    const Token& expect_kind(Tok kind) {
        if (peek().kind != kind) {
            Fail("unexpected token kind");
        }
        return advance();
    }
    const Token& expect_keyword(std::string_view value) {
        if (!is_id(peek(), value)) {
            Fail("expected keyword");
        }
        return advance();
    }
    double parse_real(const Token& tok) const {
        const std::string_view t = text(tok);
        if (options_.parse_real != nullptr) {
            return options_.parse_real(t.data(), t.size());
        }
        const std::string s(t);
        return std::strtod(s.c_str(), nullptr);
    }
    // 十进制整数；超过 kMaxBits 的值保守地失败（参考实现支持任意大整数，交给它处理）。
    std::uint64_t parse_int(const Token& tok) const {
        std::uint64_t v = 0;
        for (char c : text(tok)) {
            v = v * 10 + static_cast<std::uint64_t>(c - '0');
            if (v > kMaxBits) {
                Fail("integer too large");
            }
        }
        return v;
    }

    // ---- 语句 ----
    void parse_statement() {
        const Token& tok = peek();
        if (tok.kind == Tok::kOp && text(tok) == ";") {
            advance();
            return;
        }
        if (tok.kind != Tok::kId) {
            Fail("unexpected token");
        }
        const std::string_view kw = text(tok);
        if (kw == "include") {
            advance();
            const Token& file = expect_kind(Tok::kString);
            expect_op(";");
            if (text(file) == "\"qelib1.inc\"") {
                included_qelib_ = true;
            }
        } else if (kw == "qreg" || kw == "creg") {
            parse_register();
        } else if (kw == "gate") {
            parse_gate_declaration(false);
        } else if (kw == "opaque") {
            parse_opaque();
        } else if (kw == "barrier") {
            parse_barrier();
        } else if (kw == "measure") {
            parse_measure();
        } else if (kw == "reset" || kw == "if" || kw == "OPENQASM") {
            Fail("unsupported statement");
        } else {
            parse_gate_application();
        }
    }

    void parse_register() {
        const bool quantum = text(advance()) == "qreg";
        const Token& name_tok = expect_kind(Tok::kId);
        expect_op("[");
        const Token& size_tok = expect_kind(Tok::kInt);
        expect_op("]");
        expect_op(";");
        std::string name(text(name_tok));
        if (qregs_.count(name) > 0 || cregs_.count(name) > 0) {
            Fail("register already defined");
        }
        const std::uint64_t size = parse_int(size_tok);
        if (size == 0) {
            Fail("register size must be positive");
        }
        std::uint64_t& counter = quantum ? num_qubits_ : num_clbits_;
        if (counter + size > kMaxBits) {
            Fail("too many bits");
        }
        (quantum ? qregs_ : cregs_)
            .emplace(std::move(name), Register{static_cast<std::uint32_t>(counter), static_cast<std::uint32_t>(size)});
        counter += size;
    }

    // 与参考实现的 _check_new_gate_name 相同的重定义规则。
    void check_new_gate_name(const std::string& name, bool builtin) const {
        auto it = gates_.find(name);
        if (it == gates_.end()) {
            return;
        }
        if (it->second->builtin && !builtin && !included_qelib_ && name != "U" && name != "CX") {
            return;
        }
        Fail("gate already defined");
    }

    std::vector<std::string> parse_identifier_list() {
        std::vector<std::string> items;
        items.emplace_back(text(expect_kind(Tok::kId)));
        while (is_op(",")) {
            advance();
            items.emplace_back(text(expect_kind(Tok::kId)));
        }
        return items;
    }

    static bool has_duplicates(const std::vector<std::string>& names) {
        for (std::size_t i = 0; i < names.size(); ++i) {
            for (std::size_t j = i + 1; j < names.size(); ++j) {
                if (names[i] == names[j]) {
                    return true;
                }
            }
        }
        return false;
    }

    void parse_gate_signature(std::string& name, std::vector<std::string>& params, std::vector<std::string>& qubits) {
        name = std::string(text(expect_kind(Tok::kId)));
        params.clear();
        if (is_op("(")) {
            advance();
            if (!is_op(")")) {
                params = parse_identifier_list();
            }
            expect_op(")");
        }
        qubits = parse_identifier_list();
        if (has_duplicates(params) || has_duplicates(qubits)) {
            Fail("duplicate name in declaration");
        }
    }

    void parse_opaque() {
        expect_keyword("opaque");
        auto def = std::make_shared<GateDef>();
        std::vector<std::string> qubits;
        parse_gate_signature(def->name, def->params, qubits);
        expect_op(";");
        check_new_gate_name(def->name, false);
        def->num_qubits = static_cast<std::uint32_t>(qubits.size());
        def->opaque = true;
        gates_[def->name] = std::move(def);
    }

    static std::uint32_t formal_index(const std::string& name, const std::vector<std::string>& formals) {
        for (std::size_t i = 0; i < formals.size(); ++i) {
            if (formals[i] == name) {
                return static_cast<std::uint32_t>(i);
            }
        }
        Fail("unknown qubit name in gate declaration");
    }

    void parse_gate_declaration(bool builtin) {
        expect_keyword("gate");
        auto def = std::make_shared<GateDef>();
        std::vector<std::string> qubits;
        parse_gate_signature(def->name, def->params, qubits);
        check_new_gate_name(def->name, builtin);
        expect_op("{");
        while (!is_op("}")) {
            const Token& tok = peek();
            if (tok.kind == Tok::kEof) {
                Fail("unterminated gate body");
            }
            if (is_id(tok, "barrier")) {
                advance();
                for (const std::string& q : parse_identifier_list()) {
                    formal_index(q, qubits);
                }
                expect_op(";");
                continue;
            }
            if (is_id(tok, "measure") || is_id(tok, "reset") || is_id(tok, "if")) {
                Fail("statement not allowed inside gate");
            }
            def->body.push_back(parse_body_op(*def, qubits));
        }
        expect_op("}");
        def->num_qubits = static_cast<std::uint32_t>(qubits.size());
        def->builtin = builtin;
        gates_[def->name] = std::move(def);
    }

    BodyOp parse_body_op(GateDef& def, const std::vector<std::string>& formal_qubits) {
        const std::string callee_name(text(expect_kind(Tok::kId)));
        BodyOp op;
        if (is_op("(")) {
            advance();
            if (!is_op(")")) {
                op.params = parse_expression_list(def.pool, def.params);
            }
            expect_op(")");
        }
        const std::vector<std::string> qubit_names = parse_identifier_list();
        expect_op(";");
        for (const std::string& q : qubit_names) {
            op.qubits.push_back(formal_index(q, formal_qubits));
        }
        auto it = gates_.find(callee_name);
        if (it == gates_.end()) {
            Fail("unknown gate inside gate declaration");
        }
        check_arity(*it->second, op.params.size(), op.qubits.size());
        for (std::size_t i = 0; i < op.qubits.size(); ++i) {
            for (std::size_t j = i + 1; j < op.qubits.size(); ++j) {
                if (op.qubits[i] == op.qubits[j]) {
                    Fail("duplicate qubit argument");
                }
            }
        }
        op.gate = it->second;
        return op;
    }

    // ---- 表达式 ----
    std::int32_t node(ExprPool& pool, ExprNode n) {
        pool.push_back(n);
        return static_cast<std::int32_t>(pool.size() - 1);
    }

    std::vector<std::int32_t> parse_expression_list(ExprPool& pool, const std::vector<std::string>& formals) {
        std::vector<std::int32_t> exprs{parse_expr(pool, formals)};
        while (is_op(",")) {
            advance();
            exprs.push_back(parse_expr(pool, formals));
        }
        return exprs;
    }

    std::int32_t parse_expr(ExprPool& pool, const std::vector<std::string>& formals) {
        std::int32_t left = parse_term(pool, formals);
        while (is_op("+") || is_op("-")) {
            const bool plus = text(advance()) == "+";
            const std::int32_t right = parse_term(pool, formals);
            left = node(pool, ExprNode{plus ? ExprOp::kAdd : ExprOp::kSub, Func::kSin, 0.0, left, right});
        }
        return left;
    }

    std::int32_t parse_term(ExprPool& pool, const std::vector<std::string>& formals) {
        std::int32_t left = parse_unary(pool, formals);
        while (is_op("*") || is_op("/")) {
            const bool mul = text(advance()) == "*";
            const std::int32_t right = parse_unary(pool, formals);
            left = node(pool, ExprNode{mul ? ExprOp::kMul : ExprOp::kDiv, Func::kSin, 0.0, left, right});
        }
        return left;
    }

    std::int32_t parse_unary(ExprPool& pool, const std::vector<std::string>& formals) {
        if (is_op("-")) {
            advance();
            const std::int32_t operand = parse_unary(pool, formals);
            return node(pool, ExprNode{ExprOp::kNeg, Func::kSin, 0.0, operand, -1});
        }
        if (is_op("+")) {
            advance();
            return parse_unary(pool, formals);
        }
        return parse_power(pool, formals);
    }

    std::int32_t parse_power(ExprPool& pool, const std::vector<std::string>& formals) {
        const std::int32_t base = parse_atom(pool, formals);
        if (is_op("^")) {
            advance();
            const std::int32_t exponent = parse_unary(pool, formals);  // 右结合
            return node(pool, ExprNode{ExprOp::kPow, Func::kSin, 0.0, base, exponent});
        }
        return base;
    }

    std::int32_t parse_atom(ExprPool& pool, const std::vector<std::string>& formals) {
        const Token& tok = peek();
        if (tok.kind == Tok::kReal || tok.kind == Tok::kInt) {
            advance();
            return node(pool, ExprNode{ExprOp::kConst, Func::kSin, parse_real(tok), -1, -1});
        }
        if (tok.kind == Tok::kOp && text(tok) == "(") {
            advance();
            const std::int32_t inner = parse_expr(pool, formals);
            expect_op(")");
            return inner;
        }
        if (tok.kind == Tok::kId) {
            advance();
            const std::string_view name = text(tok);
            if (name == "pi") {
                return node(pool, ExprNode{ExprOp::kConst, Func::kSin, kPi, -1, -1});
            }
            static const std::pair<std::string_view, Func> kFuncs[] = {
                {"sin", Func::kSin}, {"cos", Func::kCos}, {"tan", Func::kTan},
                {"exp", Func::kExp}, {"ln", Func::kLn},   {"sqrt", Func::kSqrt},
            };
            for (const auto& [fname, f] : kFuncs) {
                if (name == fname) {
                    expect_op("(");
                    const std::int32_t arg = parse_expr(pool, formals);
                    expect_op(")");
                    return node(pool, ExprNode{ExprOp::kFunc, f, 0.0, arg, -1});
                }
            }
            for (std::size_t i = 0; i < formals.size(); ++i) {
                if (formals[i] == name) {
                    return node(pool, ExprNode{ExprOp::kParam, Func::kSin, 0.0, static_cast<std::int32_t>(i), -1});
                }
            }
            Fail("unknown parameter in expression");
        }
        Fail("expected an expression");
    }

    // ---- 操作数 ----
    Operand parse_operand(const std::unordered_map<std::string, Register>& registers) {
        const std::string name(text(expect_kind(Tok::kId)));
        auto it = registers.find(name);
        if (it == registers.end()) {
            Fail("unknown register");
        }
        const Register& reg = it->second;
        Operand op;
        if (!is_op("[")) {
            op.whole = true;
            op.bits.resize(reg.size);
            for (std::uint32_t i = 0; i < reg.size; ++i) {
                op.bits[i] = reg.offset + i;
            }
            return op;
        }
        advance();
        const std::uint64_t idx = parse_int(expect_kind(Tok::kInt));
        expect_op("]");
        if (idx >= reg.size) {
            Fail("index out of range");
        }
        op.whole = false;
        op.bits.push_back(reg.offset + static_cast<std::uint32_t>(idx));
        return op;
    }

    std::vector<Operand> parse_qubit_operands() {
        std::vector<Operand> ops;
        ops.push_back(parse_operand(qregs_));
        while (is_op(",")) {
            advance();
            ops.push_back(parse_operand(qregs_));
        }
        return ops;
    }

    // ---- 量子操作 ----
    void append(const Operation& op, const std::uint32_t* qubits, std::size_t nq, const std::uint32_t* clbits,
                std::size_t nc) {
        out_->append(op, qubits, nq, clbits, nc);
        for (std::size_t i = 0; i < nq; ++i) {
            if (qubits[i] >= active_.size()) {
                active_.resize(static_cast<std::size_t>(qubits[i]) + 1, 0);
            }
            active_[qubits[i]] = 1;
        }
    }

    void parse_barrier() {
        expect_keyword("barrier");
        std::vector<std::uint32_t> qubits;
        if (is_op(";")) {
            advance();
            // 不带参数的 barrier：当前所有用到的比特（升序）。
            for (std::size_t q = 0; q < active_.size(); ++q) {
                if (active_[q] != 0) {
                    qubits.push_back(static_cast<std::uint32_t>(q));
                }
            }
        } else {
            for (const Operand& op : parse_qubit_operands()) {
                for (std::uint32_t b : op.bits) {
                    bool seen = false;
                    for (std::uint32_t q : qubits) {
                        seen = seen || q == b;
                    }
                    if (!seen) {
                        qubits.push_back(b);
                    }
                }
            }
            expect_op(";");
        }
        Operation op;
        op.kind = OpKind::kBarrier;
        append(op, qubits.data(), qubits.size(), nullptr, 0);
    }

    void parse_measure() {
        expect_keyword("measure");
        const Operand q = parse_operand(qregs_);
        expect_op("->");
        const Operand c = parse_operand(cregs_);
        expect_op(";");
        if (q.whole != c.whole || q.bits.size() != c.bits.size()) {
            Fail("measure operand mismatch");
        }
        Operation op;
        op.kind = OpKind::kMeasure;
        for (std::size_t i = 0; i < q.bits.size(); ++i) {
            append(op, &q.bits[i], 1, &c.bits[i], 1);
        }
    }

    void parse_gate_application() {
        const std::string name(text(expect_kind(Tok::kId)));
        std::vector<double> params;
        if (is_op("(")) {
            advance();
            if (!is_op(")")) {
                ExprPool pool;
                static const std::vector<std::string> kNoFormals;
                for (std::int32_t e : parse_expression_list(pool, kNoFormals)) {
                    params.push_back(Eval(pool, e, nullptr));
                }
            }
            expect_op(")");
        }
        const std::vector<Operand> operands = parse_qubit_operands();
        expect_op(";");
        auto it = gates_.find(name);
        if (it == gates_.end()) {
            Fail("unknown gate");
        }
        const GateDef& gate = *it->second;
        check_arity(gate, params.size(), operands.size());
        // 整寄存器广播。
        std::size_t size = 0;
        bool any_whole = false;
        for (const Operand& op : operands) {
            if (op.whole) {
                if (any_whole && op.bits.size() != size) {
                    Fail("register arguments must have the same size");
                }
                size = op.bits.size();
                any_whole = true;
            }
        }
        const std::size_t reps = any_whole ? size : 1;
        std::vector<std::uint32_t> qubits(operands.size());
        for (std::size_t i = 0; i < reps; ++i) {
            for (std::size_t k = 0; k < operands.size(); ++k) {
                qubits[k] = operands[k].whole ? operands[k].bits[i] : operands[k].bits[0];
            }
            for (std::size_t a = 0; a < qubits.size(); ++a) {
                for (std::size_t b = a + 1; b < qubits.size(); ++b) {
                    if (qubits[a] == qubits[b]) {
                        Fail("duplicate qubit argument");
                    }
                }
            }
            apply(gate, params.data(), params.size(), qubits.data(), 0);
        }
    }

    static void check_arity(const GateDef& gate, std::size_t num_params, std::size_t num_qubits) {
        if (num_params != gate.params.size() || num_qubits != gate.num_qubits) {
            Fail("arity mismatch");
        }
    }

    void apply(const GateDef& gate, const double* params, std::size_t num_params, const std::uint32_t* qubits,
               int depth) {
        if (gate.native) {
            Operation op;
            op.kind = *gate.native;
            op.num_params = static_cast<std::uint8_t>(num_params);
            for (std::size_t i = 0; i < num_params; ++i) {
                op.params[i] = params[i];
            }
            append(op, qubits, gate.num_qubits, nullptr, 0);
            return;
        }
        if (gate.opaque) {
            Fail("opaque gate cannot be imported");
        }
        if (depth >= kMaxExpansionDepth) {
            Fail("gate definitions nested too deeply");
        }
        double values[16];
        std::uint32_t mapped[64];
        for (const BodyOp& op : gate.body) {
            if (op.params.size() > 16 || op.qubits.size() > 64) {
                Fail("gate call too large");
            }
            for (std::size_t i = 0; i < op.params.size(); ++i) {
                values[i] = Eval(gate.pool, op.params[i], params);
            }
            for (std::size_t i = 0; i < op.qubits.size(); ++i) {
                mapped[i] = qubits[op.qubits[i]];
            }
            apply(*op.gate, values, op.params.size(), mapped, depth + 1);
        }
    }

    static constexpr double kPi = 3.141592653589793;  // math.pi
    // 用户 gate 的嵌套展开深度上限：参考实现在 Python 递归上限附近会抛 RecursionError，
    // 远低于该处就交还给参考实现。
    static constexpr int kMaxExpansionDepth = 100;

    std::string_view src_;
    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
    GateTable gates_;
    const Qasm2Options& options_;
    Circuit* out_;
    std::unordered_map<std::string, Register> qregs_;
    std::unordered_map<std::string, Register> cregs_;
    std::uint64_t num_qubits_ = 0;
    std::uint64_t num_clbits_ = 0;
    bool included_qelib_ = false;
    std::vector<std::uint8_t> active_;
};

}  // namespace

class Qasm2Library {
public:
    GateTable gates;
};

std::shared_ptr<const Qasm2Library> CompileQasm2Library(std::string_view definitions, const Qasm2Options& options) {
    GateTable table;
    for (const NativeGate& g : kNativeGates) {
        auto def = std::make_shared<GateDef>();
        def->name = g.qasm_name;
        for (std::uint32_t i = 0; i < g.num_params; ++i) {
            def->params.push_back("p" + std::to_string(i));
        }
        def->num_qubits = g.num_qubits;
        def->builtin = true;
        def->native = g.kind;
        table[def->name] = std::move(def);
    }
    try {
        Parser parser(definitions, std::move(table), options, nullptr);
        parser.parse_library();
        auto lib = std::make_shared<Qasm2Library>();
        lib->gates = parser.take_gates();
        return lib;
    } catch (const Reject& r) {
        throw std::invalid_argument("invalid OpenQASM gate library: " + r.reason);
    }
}

bool ParseQasm2(std::string_view source, const Qasm2Library& library, Circuit& out, std::string* error,
                const Qasm2Options& options) {
    try {
        Parser parser(source, library.gates, options, &out);
        parser.parse_program();
        return true;
    } catch (const Reject& r) {
        if (error != nullptr) {
            *error = r.reason;
        }
        out.clear();
        return false;
    }
}

}  // namespace quspect::native::ir
