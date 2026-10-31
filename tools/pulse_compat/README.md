# pulse_compat：quspect._pulse 与 envelopes-qc 0.1.5 的差分比对

`quspect._pulse` 是 envelopes-qc 0.1.5 C++ 扩展（`envelopes.envelopes_cpp`）的重新实现，
目标是接口与行为一致、解码波形逐位相同。本目录的工具在两边分别运行同一批用例并逐项比较：
波形按原始字节比较，另外比较序列化字节、pickle、repr、异常类型与消息、警告类别等；
模块路径差异（`quspect._pulse` ↔ `envelopes.envelopes_cpp`）会先归一化。

```bash
# 需要一个装有 envelopes-qc==0.1.5 的解释器作为参考实现
uv venv /tmp/ref && uv pip install -p /tmp/ref envelopes-qc==0.1.5
.venv/bin/python tools/pulse_compat/compare.py --ref-python /tmp/ref/bin/python
COMPAT_EXPRS=3000 .venv/bin/python tools/pulse_compat/compare.py --ref-python /tmp/ref/bin/python
```

- `compat_cases.py`：用例目录（每种包络、随机组合表达式、运算符、split、窗口解码、Custom 输入、
  序列化等）；`--module` 指定被测模块。
- `compare.py`：运行两边并输出差异；有差异时退出码非 0。
- `make_golden.py`：用参考实现重新生成 `tests/pulse/golden.json`（CI 中不需要参考实现）。

原实现在以下输入上会段错误或卡死，这里改为正常结果或抛出异常，因此不在比对范围内：
空 `EnvSum` 的 `start()`/解码、对“实数乘积包着复数包络”调用 `mix()`/`copy()`、`s += s`、
把 `None` 当作包络传入、截断的序列化数据。
