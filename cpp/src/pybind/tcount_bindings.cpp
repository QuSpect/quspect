// T 门数优化搜索内核的 Python 绑定（见 quspect_native/transpiler/tcount.hpp）。
// 只做数据搬运：列是 64 位掩码，问题按元组给出；计算期间释放 GIL。

#include "tcount_bindings.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quspect_native/common/parallel.hpp"
#include "quspect_native/transpiler/tcount.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using Columns = std::vector<std::uint64_t>;
using ColumnProblem = std::tuple<Columns, int, int, std::uint64_t>;
using Term = std::array<std::uint64_t, 3>;
using FlipProblem = std::tuple<std::vector<Term>, std::int64_t, std::uint64_t>;

}  // namespace

void BindTCount(py::module_& m) {
    m.def(
        "_tcount_todd_batch",
        [](const std::vector<ColumnProblem>& problems) {
            std::vector<transpiler::TCountToddProblem> probs;
            probs.reserve(problems.size());
            for (const auto& [cols, k, plateau, seed] : problems)
                probs.push_back(transpiler::TCountToddProblem{cols, k, plateau, seed});
            std::vector<Columns> out;
            {
                py::gil_scoped_release release;
                out = transpiler::TCountToddBatch(probs, common::CompileThreads());
            }
            return out;
        },
        py::arg("problems"),
        "TODD on phase-polynomial column sets: problems are (columns, k, plateau, seed); list of reduced column lists.");

    m.def(
        "_tcount_rm_search",
        [](const std::vector<ColumnProblem>& problems, int samples) {
            std::vector<transpiler::TCountRmProblem> probs;
            probs.reserve(problems.size());
            for (const auto& [cols, k, iterations, seed] : problems)
                probs.push_back(transpiler::TCountRmProblem{cols, k, iterations, seed});
            std::vector<Columns> out;
            {
                py::gil_scoped_release release;
                out = transpiler::TCountRmSearch(probs, samples, common::CompileThreads());
            }
            return out;
        },
        py::arg("problems"), py::arg("samples"),
        "Reed-Muller flat search: problems are (columns, k, iterations, seed); list of the best column lists.");

    m.def(
        "_tcount_flip_walks",
        [](const std::vector<FlipProblem>& problems, int plus_after) {
            std::vector<transpiler::TCountFlipProblem> probs;
            probs.reserve(problems.size());
            for (const auto& [start, steps, seed] : problems)
                probs.push_back(transpiler::TCountFlipProblem{start, steps, seed});
            std::vector<std::vector<Term>> out;
            {
                py::gil_scoped_release release;
                out = transpiler::TCountFlipWalks(probs, plus_after, common::CompileThreads());
            }
            return out;
        },
        py::arg("problems"), py::arg("plus_after"),
        "Flip-graph walks on trilinear decompositions: problems are (terms, steps, seed); lowest-rank terms seen.");
}

}  // namespace quspect::native::pybind
