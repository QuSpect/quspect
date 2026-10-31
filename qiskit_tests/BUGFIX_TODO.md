# Bug 修复 Todo List

来源：`qiskit_tests` 中全部 `xfail(strict=True)` 用例（共 83 条，按“测试文件 + 原因”去重），另有 2 条附加项。

**完成判据（机械可验证）**

1. `grep -rn "xfail" qiskit_tests/test_*.py` 没有任何结果。
2. `pytest qiskit_tests` 结果为 0 failed、0 xfailed。
3. `pytest`（原有 `tests/`）全部通过；`ruff check`、`ruff format --check`、`mypy quspect` 零错误。
4. C++ 改动需要通过 ctest，并经过 clang-format 格式化。

状态标记：`[ ]` 未开始，`[x]` 已修复并已去掉 xfail，`[!]` 有异议（测试本身的期望值有误，需要说明理由）。

---

## WP1 circuit / gates / statevector（`circuit/*`、`gates/*`、`simulation/simulator.py`、`simulation/gate_applicator.py`）

- [x] B05 `cx(0,0)` 等门的 qubit 重复时应报错 — test_circuit_api::test_duplicate_qubits_in_gate_rejected
- [x] B06 `CircuitInstruction.matrix()` 遇到符号参数应抛 ValueError — test_circuit_api::test_instruction_matrix_symbolic_raises_value_error
- [x] B07 `compose()` 丢失噪声 channel — test_circuit_api::test_compose_preserves_noise_channels
- [x] B08 `to_json()` 丢失 channel 强度 — test_circuit_json::test_json_roundtrip_preserves_channel_strength
- [x] B09 JSON 往返后含噪线路的分布与 qiskit 不一致 — test_circuit_json::test_json_roundtrip_noisy_distribution_matches_qiskit
- [x] B10 `gen_amp_damp` 的 JSON 名称不一致 — test_circuit_json::test_json_roundtrip_generalized_amplitude_damping
- [x] B11 `bind_parameters` 没有校验值的类型 — test_circuit_parameters::test_bind_parameters_rejects_non_real_value
- [x] B12 numpy 标量角度不能序列化 — test_circuit_parameters::test_numpy_scalar_angles_are_simulable
- [x] B26 `gate_matrix` 遇到多余参数时应报错 — test_gates_api::test_gate_matrix_rejects_extra_params
- [x] B71 空线路的 `statevector()` 应返回 `[1]` — test_simulation_statevector::test_statevector_empty_circuit_is_unit_scalar
- [x] B72 含中途测量时 `statevector()` 返回空数组 — test_simulation_statevector::test_statevector_mid_circuit_measurement_is_valid_or_raises
- [x] B73 `apply_gate` 应拒绝长度不是 2 的幂的态 — test_simulation_statevector::test_apply_gate_rejects_non_power_of_two_state
- [x] B74 `apply_gate` 遇到非 complex128 或非连续数组时静默不生效 — test_simulation_statevector::test_apply_gate_non_complex128_state_updated_or_rejected
- [x] X2a 消除既有的 ruff 错误：`simulation/gate_applicator.py`、`tests/circuit/test_circuit.py`

## WP2 噪声 / 密度矩阵（`noise/*`、`simulation/density_matrix.py`、`simulation/density_simulator.py`）

- [x] B01 DMS（`DensityMatrixSimulator`）经 backend 执行时忽略 clbit 映射 — test_backends_core::test_density_matrix_backend_partial_measure_uses_clbits
- [x] B13 `DensityMatrix.probabilities` 对子集取边缘概率时，把测量的比特迹掉了 — test_conventions::test_density_matrix_probabilities_subset_bug
- [x] B20 `subsystem_purity` 算成了补集的纯度 — test_density_matrix::test_subsystem_purity_is_on_given_qubits
- [x] B21 `probabilities` 的子集或乱序结果错误 — test_density_matrix::test_probabilities_subset_and_order_matches_qiskit
- [x] B22 `measure` 对子集测量时测错了比特 — test_density_matrix::test_measure_subset_collapses_correct_qubit
- [x] B23 `measure` 在比特顺序打乱时结果错误 — test_density_matrix::test_measure_permuted_qubits
- [x] B24 DMS.run 忽略 clbit 映射 — test_density_simulator::test_run_keys_follow_clbits_like_simulator
- [x] B25 DMS.run 的中途测量分布错误 — test_density_simulator::test_run_mid_circuit_measurement_distribution
- [x] B48 单比特 channel 应拒绝 `num_qubits≠1` — test_noise_channels::test_single_qubit_channels_reject_wrong_width
- [x] B49 KrausChannel 的 docstring 示例不保迹 — test_noise_channels::test_kraus_channel_docstring_example_constructs
- [x] B50 KrausChannel 判等时忽略了算符 — test_noise_channels::test_kraus_channel_equality_considers_operators
- [x] B51 1 比特 channel 作用于多比特门时，维度不匹配 — test_noise_model::test_adder_channel_arity_matches_qubits
- [x] B52 T2 公式错误 — test_noise_model::test_thermal_relaxation_matches_t1_t2_formula
- [x] B53 标量 T1/T2 只作用于 qubit 0 — test_noise_model::test_thermal_relaxation_scalar_applies_to_all_qubits
- [x] B54 `excited_state_population` 被忽略 — test_noise_model::test_thermal_relaxation_excited_state_population
- [x] B55 `thermal_relaxation_model` 作用于 2 比特门时出错 — test_noise_model::test_thermal_relaxation_model_two_qubit_gate
- [x] B56 `add_readout_error` 从不生效 — test_noise_model::test_readout_error_inserted_before_measurement
- [x] B75 `thermal_relaxation_model` 下的 Bell 轨迹 — test_trajectory_noise::test_thermal_relaxation_model_bell_trajectories
- [x] X2b 消除 `simulation/density_matrix.py` 中既有的 ruff 和 mypy 错误

## WP3 DAG / passes / transpiler（`dag/*`、`transpiler/*`）

- [x] B14 `PhasePolynomial.extract` 中 `cx(1,0)` 的方向错误 — test_dag_phase_polynomial::test_extract_reversed_cnot_bug
- [x] B15 `extract` 中 CZ 的相位错误 — test_dag_phase_polynomial::test_extract_cz_phase_bug
- [x] B16 `extract` 的一般情况与 numpy 结果不符 — test_dag_phase_polynomial::test_extract_matches_numpy_general_bug
- [x] B17 `to_circuit` 合成结果错误 — test_dag_phase_polynomial::test_to_circuit_general_bug
- [x] B18 `optimize()` 破坏线路语义 — test_dag_phase_polynomial::test_extract_and_optimize_preserves_semantics_bug
- [x] B19 `DAGToCircuitCompiler` 丢失 channel — test_dag_structure::test_roundtrip_preserves_noise_channels
- [x] B69 PassManager 在 commute 之后没有再做一次 cancel — test_passes_optimization::test_cancel_then_commute_pipeline_recancels_bug
- [x] B70 PassManager 不收敛时应抛 RuntimeError — test_passes_optimization::test_pass_manager_nonconvergence_raises_as_documented
- [x] B76 SABRE 遇到不连通拓扑时应快速失败 — test_transpiler_sabre::test_sabre_disconnected_components_fail_fast_bug
- [x] B77 OptimizePass 消不掉 cx-rz-cx — test_transpiler_target_validation::test_optimize_pass_commutation_cancellation_bug
- [x] X2c 消除 `dag/passes/gate_cancellation.py` 中既有的 mypy 错误

## WP4 QASM2 导入导出（`ir/qasm_importer.py`、`ir/qasm2.py`、`ir/compilation.py`）

- [x] B27 导出浮点数只保留 10 位有效数字 — test_ir_qasm2_export::test_float_parameters_roundtrip_exactly
- [x] B28 DECOMPOSE 模式只分解一层 — test_ir_qasm2_export::test_decompose_mode_lowers_nested_custom_rules
- [x] B29 EXTENDED 模式导出带参数的自定义门时生成非法 QASM — test_ir_qasm2_export::test_extended_mode_parameterized_custom_gate
- [x] B30 EXTENDED 模式静默丢掉子门 — test_ir_qasm2_export::test_extended_mode_does_not_drop_unknown_body_gates
- [x] B31 qelib1 标准门报 Unknown gate — test_ir_qasm2_import::test_unsupported_qelib1_gates
- [x] B32 参数表达式（如 `pi/2`、`sin`）无法解析 — test_ir_qasm2_import::test_parameter_expressions
- [x] B33 qiskit 导出的 `pi/2` 无法读回 — test_ir_qasm2_import::test_qiskit_dump_with_pi_fraction_angles
- [x] B34 `.5` 被解析成 5 — test_ir_qasm2_import::test_leading_dot_float_literal
- [x] B35 没有校验门的参数个数 — test_ir_qasm2_import::test_wrong_parameter_count_raises
- [x] B36 多个寄存器相互别名 — test_ir_qasm2_import::test_multiple_registers_concatenate_like_qiskit
- [x] B37 不支持整寄存器广播 — test_ir_qasm2_import::test_register_broadcast
- [x] B38 带形参的自定义门无法解析 — test_ir_qasm2_import::test_parameterized_custom_gate
- [x] B39 自定义门不能嵌套调用 — test_ir_qasm2_import::test_nested_custom_gates
- [x] B40 `if` 语句被静默丢弃 — test_ir_qasm2_import::test_if_statement_not_silently_dropped
- [x] B41 QASM2ParseError 不带行号 — test_ir_qasm2_import::test_parse_error_reports_line_number
- [x] B42 非法字符被静默跳过 — test_ir_qasm2_import::test_invalid_characters_raise

## WP5 operators / optimization / variational / backends（`operators/*`、`optimization/*`、`variational/*`、`backends/*`）

- [x] B02 `workload_id` 应在提交之前校验 — test_backends_logicalqubit::test_workload_id_validated_before_submission
- [x] B03 `result(timeout=nan)` 应被拒绝 — test_backends_logicalqubit::test_job_result_rejects_nan_timeout
- [x] B04 `return_batch=True` 且不传 labels 时构造失败 — test_backends_logicalqubit::test_job_return_batch_without_labels
- [x] B57 `hopping_term(i,i)` 的结果少了一半 — test_operators_fermion::test_hopping_term_same_mode_is_twice_number_operator
- [x] B58 `normal_order` 中 delta 项的符号错误 — test_operators_fermion::test_normal_order_delta_sign_after_previous_swap
- [x] B59 `normal_order` 改变了算符的值 — test_operators_fermion::test_normal_order_preserves_operator_value_random
- [x] B60 Fermion 的 LaTeX 输出没有省略 ±1 — test_operators_fermion::test_latex_omits_unit_coefficients
- [x] B61 QubitOperator 没有校验 Pauli 字母 — test_operators_qubit::test_constructor_rejects_invalid_pauli_letter
- [x] B62 QubitOperator 中同一个 qubit 重复出现 — test_operators_qubit::test_constructor_repeated_qubit_is_rejected_or_reduced
- [x] B63 Qubit 的 LaTeX 输出没有省略 ±1 — test_operators_qubit::test_latex_omits_unit_coefficients
- [x] B64 BK 变换违反 CAR — test_operators_transforms::test_bk_canonical_anticommutation
- [x] B65 BK 下 {a0,a2} 不为零 — test_operators_transforms::test_bk_minimal_anticommutator_repro
- [x] B66 JW 与 BK 的谱不一致 — test_operators_transforms::test_jw_bk_spectra_agree
- [x] B67 BQM 对称化时溢出 — test_optimization_conversion::test_bqm_large_finite_quadratic_stays_finite
- [x] B68 HardwareIsingProgram 对称化时溢出 — test_optimization_hardware::test_hardware_program_large_finite_matrix_stays_finite
- [x] B78 AdjointEngine 没有校验非有限角度 — test_variational_adjoint::test_adjoint_rejects_non_finite_gate_angles
- [x] X2d 消除 `backends/logicalqubit/{_job,_provider,_translator}.py` 中既有的 mypy 错误

## WP6 C++ native / QSAL / 可视化（`cpp/*`、`ir/qsal.py`、`native/*`、`visualization/*`）

- [x] B43 QSAL 观测量的复系数丢失虚部 — test_ir_qsal::test_observable_complex_coefficient_roundtrip
- [x] B44 QSAL 没有保存线路名 — test_ir_qsal::test_circuit_name_roundtrip
- [x] B45 QSAL 没有保存 Gate.label — test_ir_qsal::test_gate_label_roundtrip
- [x] B46 C++ 中多个 X/Y 因子的期望值错误 — test_ir_qsal::test_native_expectation_multi_flip_terms_match_qiskit
- [x] B47 C++ 中 ⟨XX⟩、⟨YY⟩ 在 \|++⟩ 上的结果错误 — test_ir_qsal::test_native_expectation_xx_on_plus_plus
- [x] X1 AVX2 期望值内核中同样的 `j <= i` 重复去重（`cpp/src/kernels/avx2/expectation.cpp:111`）
- [x] B79 SamplingStats 把 0 计数也算作已观测 — test_visualization_sampling::test_stats_ignore_unobserved_outcomes
- [x] B80 VBAR 的柱与标签错位 — test_visualization_sampling::test_vbar_bars_align_with_labels
- [x] B81 0 shots 时 DOT 图除零 — test_visualization_sampling::test_zero_shot_dot_chart
- [x] B82 SVG 中受控旋转门不显示角度 — test_visualization_svg::test_controlled_rotation_label_shows_angle
- [x] B83 文本图中受控旋转门不显示角度 — test_visualization_text::test_controlled_rotation_shows_angle
