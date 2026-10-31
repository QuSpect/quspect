Architecture
============

This page describes the high-level architecture of QuSpect.

Layered Architecture
--------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    User API Layer                        │
   │   QuantumCircuit, Gate, Parameter                       │
   ├─────────────────────────────────────────────────────────┤
   │                  Core Data Model                         │
   │   CircuitData, DAGCircuit, DAGNode, DAGEdge            │
   ├─────────────────────────────────────────────────────────┤
   │                   Compilation Layer                      │
   │   CircuitCompiler (Circuit → DAG)                       │
   └─────────────────────────────────────────────────────────┘

Core Concepts
-------------

Gate System
~~~~~~~~~~~

Per architecture decision 7.1, gate definitions are separated from matrix
computation (:ref:`方案 C <architecture-decision-7-1>`):

- :class:`~quspect.gates.gate.Gate` — Lightweight metadata container
  (name, num_qubits, label). Hashable, serializable, no numpy dependency.
- :func:`~quspect.gates._matrix_utils.get_gate_matrix` — Central dispatch
  for matrix computation. Handles both standard and parameterized gates.

Circuit System
~~~~~~~~~~~~~~

:class:`~quspect.circuit.circuit.QuantumCircuit` uses dynamic qubit/clbit
count inference (architecture decision 7.0). No need to pre-declare circuit
size — it grows automatically as qubits are referenced.

Measurements are represented as :class:`~quspect.circuit.instruction.InstructionType`
rather than as Gate subclasses (architecture decision 7.3), since measurement
is a projective operation, not a unitary transformation.

DAG System
~~~~~~~~~~

The Directed Acyclic Graph (DAG) is QuSpect's primary intermediate
representation. The compiler converts sequential circuit instructions
into a dependency graph that explicitly models qubit/clbit data flow.

- Time complexity: O(n + m) where n = num_qubits, m = instructions
- Supports topological ordering and layer partitioning for scheduling

Architecture Decisions
----------------------

.. _architecture-decision-7-0:

7.0: Dynamic Qubit Count
~~~~~~~~~~~~~~~~~~~~~~~~

**Decision**: ``QuantumCircuit()`` does not accept a qubit count parameter.
``num_qubits`` and ``num_clbits`` are inferred from actual usage
(``max(index) + 1``).

.. _architecture-decision-7-1:

7.1: Gate — Definition vs. Matrix Separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision**: 方案 C. Gate is a pure metadata container. Matrix computation
is handled by standalone functions in ``_matrix_utils.py``. This keeps
gate definitions lightweight, hashable, and free from numpy serialization
issues.

7.2: Instruction Storage
~~~~~~~~~~~~~~~~~~~~~~~~

**Decision**: 方案 A. Simple ``list[CircuitInstruction]`` for Phase 1.
Indexed access can be added later if performance requires.

7.3: Measurement Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision**: 方案 B. Measurement is an ``InstructionType``, not a Gate
subclass. This correctly models the fact that measurement is a projective
(non-unitary) operation.

7.4: Parameter Storage
~~~~~~~~~~~~~~~~~~~~~~

**Decision**: 方案 B. ``params: tuple[float | Parameter, ...]`` —
supports both concrete numeric values and symbolic parameters.
